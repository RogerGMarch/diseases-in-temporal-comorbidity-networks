"""
Robustness Analysis Module.

This script strictly reproduces the logic and visualizations of '008_RobustnessTest.ipynb'.
It performs a sensitivity analysis by comparing results at OR > 1.5 (Baseline) vs OR > 2.0 (Strict).

Methodology:
1. Load Data (ICD, Prevalence, Mortality, Adjacency Matrices).
2. For each threshold:
    a. Binarize matrix -> Compute Topology (Degree/Betweenness).
    b. Identify Outliers (80th percentile Log(Degree/Prevalence)).
    c. Identify Sinks (80th percentile Z-Score Product of Betweenness * Mortality).
    d. Identify Bridges (95th percentile Z-Score Product of Edge Betw * Mort Diff).
3. Compute Overlap (Jaccard Index).
4. Extract Edge Weights for the identified components (Baseline only).
5. Generate Publication-Ready Figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import typer
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from tapas.config import (
    PROCESSED_DATA_DIR, 
    FIGURES_DIR, 
    SEXES, 
    AGE_GROUPS,
    DATA_DIR,
    INTERIM_DATA_DIR
)

app = typer.Typer()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

THRESHOLDS = {
    'or_1.5': {'name': 'OR > 1.5 (Baseline)', 'value': 1.5},
    'or_2.0': {'name': 'OR > 2.0 (Strict)', 'value': 2.0}
}

# Specific colors from the notebook
COLORS = {
    'Outliers': '#3498db',  # Blue
    'Sinks': '#e74c3c',     # Red
    'Bridges': '#2ecc71'    # Green
}

ROBUSTNESS_FIG_DIR = FIGURES_DIR / "robustness"
ROBUSTNESS_FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

def load_icd_mapping() -> pd.DataFrame:
    """Load ICD mapping. Returns DataFrame with 'diagnose_id' and 'icd_code'."""
    # Try extracted path first, then raw
    path = INTERIM_DATA_DIR / "ICD10_Diagnoses.csv"
    if not path.exists():
        path = DATA_DIR / "ICD10_Diagnoses.csv"
    
    df = pd.read_csv(path)
    # Standardize columns
    if 'Id' in df.columns:
        df = df.rename(columns={'Id': 'diagnose_id', 'Code': 'icd_code'})
    return df

def load_prevalence() -> pd.DataFrame:
    """Load prevalence data."""
    path = INTERIM_DATA_DIR / "extracted" / "Data" / "1.Prevalence" / "Prevalence_Sex_Age_Year_ICD.csv"
    if not path.exists():
        path = DATA_DIR / "1.Prevalence" / "Prevalence_Sex_Age_Year_ICD.csv"
    return pd.read_csv(path)

def load_mortality() -> pd.DataFrame:
    """Load and combine mortality data for Male and Female."""
    dfs = []
    for sex in SEXES:
        filename = f"mortality_diag_{sex}.csv"
        path = INTERIM_DATA_DIR / "extracted" / "Data" / filename
        if not path.exists():
            path = DATA_DIR / filename
        
        if path.exists():
            df = pd.read_csv(path)
            df['sex'] = sex
            dfs.append(df)
            
    if not dfs:
        logger.error("Could not load mortality files.")
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    # Ensure numeric
    if 'mortality' in combined.columns:
        combined['mortality'] = pd.to_numeric(combined['mortality'], errors='coerce').fillna(0)
    return combined

def load_adjacency_matrix(sex: str, age_group: int) -> Optional[np.ndarray]:
    """Load raw adjacency matrix (weighted)."""
    filename = f"Adj_Matrix_{sex}_ICD_age_{age_group}.csv"
    path = INTERIM_DATA_DIR / "extracted" / "Data" / "3.AdjacencyMatrices" / filename
    if not path.exists():
        path = DATA_DIR / "3.AdjacencyMatrices" / filename
    
    if path.exists():
        try:
            return pd.read_csv(path, sep=' ', header=None).values
        except Exception as e:
            logger.warning(f"Failed to load matrix {path}: {e}")
            return None
    return None

# ==============================================================================
# CORE ALGORITHMS (REPLICATING NOTEBOOK LOGIC)
# ==============================================================================

def compute_degree_outliers(
    sex: str, 
    age_group: int, 
    threshold: float,
    icd_df: pd.DataFrame,
    prev_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Replicates 'compute_degree_outliers' from Notebook Cell 21.
    Logic:
    1. Load Matrix, Binarize by Threshold.
    2. Compute Degree.
    3. Calculate Ratio = Degree / Prevalence.
    4. Outliers = Top 20% (80th percentile) of Log10(Ratio).
    """
    A = load_adjacency_matrix(sex, age_group)
    if A is None: return pd.DataFrame()
    
    # Apply Threshold to create Binary Graph for topology
    A_binary = (A >= threshold).astype(float)
    G = nx.from_numpy_array(A_binary)
    
    nodes_data = []
    # diagnose_id in CSV is 1-based, matrix index is 0-based
    icd_map = dict(zip(icd_df['diagnose_id'] - 1, icd_df['icd_code']))
    
    for node_idx in range(len(A)):
        degree = G.degree(node_idx)
        if degree > 0:
            icd_code = icd_map.get(node_idx)
            if icd_code:
                prev = prev_dict.get(icd_code, 0)
                if prev > 0:
                    ratio = degree / prev
                    log_ratio = np.log10(ratio)
                    nodes_data.append({
                        'node': node_idx,
                        'icd_code': icd_code,
                        'degree': degree,
                        'prevalence': prev,
                        'log_ratio': log_ratio
                    })
    
    df = pd.DataFrame(nodes_data)
    if not df.empty:
        # 80th Percentile Rule
        upper_bound = df['log_ratio'].quantile(0.80)
        outliers = df[df['log_ratio'] >= upper_bound].copy()
        outliers['Sex'] = sex
        outliers['Age_Group'] = age_group
        return outliers
        
    return pd.DataFrame()

def compute_high_mortality_sinks(
    sex: str, 
    age_group: int, 
    threshold: float,
    icd_df: pd.DataFrame,
    mort_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Replicates 'compute_high_mortality_sinks' from Notebook Cell 21.
    Logic:
    1. Load Matrix, Binarize by Threshold.
    2. Compute Betweenness Centrality on BINARY graph.
    3. Z-score Betweenness & Z-score Mortality.
    4. Sinks = Top 20% (80th percentile) of Product (Z_bet * Z_mort).
    """
    A = load_adjacency_matrix(sex, age_group)
    if A is None: return pd.DataFrame()
    
    # Apply Threshold -> Binary Graph
    A_binary = (A >= threshold).astype(float)
    G = nx.from_numpy_array(A_binary)
    
    # Calculate Betweenness on Binary Graph (Crucial for reproducing results)
    betweenness = nx.betweenness_centrality(G)
    
    nodes_data = []
    icd_map = dict(zip(icd_df['diagnose_id'] - 1, icd_df['icd_code']))
    
    for node_idx in range(len(A)):
        bet = betweenness.get(node_idx, 0)
        if bet > 0:
            icd_code = icd_map.get(node_idx)
            if icd_code:
                mort = mort_dict.get(icd_code, 0)
                nodes_data.append({
                    'node': node_idx,
                    'icd_code': icd_code,
                    'betweenness': bet,
                    'mortality': mort
                })
                
    df = pd.DataFrame(nodes_data)
    if not df.empty:
        # Z-Score Calculation (matches notebook cell 21 logic)
        mean_bet, std_bet = df['betweenness'].mean(), df['betweenness'].std()
        mean_mort, std_mort = df['mortality'].mean(), df['mortality'].std()
        
        if std_bet > 0 and std_mort > 0:
            df['z_betweenness'] = (df['betweenness'] - mean_bet) / std_bet
            df['z_mortality'] = (df['mortality'] - mean_mort) / std_mort
            df['z_product'] = df['z_betweenness'] * df['z_mortality']
            
            # 80th Percentile Rule on Product
            thresh_val = df['z_product'].quantile(0.80)
            sinks = df[df['z_product'] >= thresh_val].copy()
            sinks['Sex'] = sex
            sinks['Age_Group'] = age_group
            return sinks
            
    return pd.DataFrame()

def compute_high_mortality_bridges(
    sex: str,
    age_group: int,
    threshold: float,
    icd_df: pd.DataFrame,
    mort_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Replicates 'compute_high_mortality_bridges'.
    Logic: Edge Betweenness Z-score * Mortality Diff Z-score -> Top 5%.
    """
    A = load_adjacency_matrix(sex, age_group)
    if A is None: return pd.DataFrame()
    
    A_binary = (A >= threshold).astype(float)
    G = nx.from_numpy_array(A_binary)
    
    edge_betweenness = nx.edge_betweenness_centrality(G)
    icd_map = dict(zip(icd_df['diagnose_id'] - 1, icd_df['icd_code']))
    
    edges_data = []
    
    for (u, v), bet in edge_betweenness.items():
        if bet > 0:
            icd1 = icd_map.get(u)
            icd2 = icd_map.get(v)
            if icd1 and icd2:
                mort1 = mort_dict.get(icd1, 0)
                mort2 = mort_dict.get(icd2, 0)
                mort_diff = abs(mort1 - mort2)
                
                edges_data.append({
                    'node1': u, 'node2': v,
                    'icd1': icd1, 'icd2': icd2,
                    'betweenness': bet,
                    'mort_diff': mort_diff
                })
                
    df = pd.DataFrame(edges_data)
    if not df.empty:
        # Filter: Diff >= 0.10 (Hardcoded in notebook)
        df = df[df['mort_diff'] >= 0.10].copy()
        
        if len(df) > 0:
            mean_bet, std_bet = df['betweenness'].mean(), df['betweenness'].std()
            mean_diff, std_diff = df['mort_diff'].mean(), df['mort_diff'].std()
            
            if std_bet > 0 and std_diff > 0:
                df['z_betweenness'] = (df['betweenness'] - mean_bet) / std_bet
                df['z_mort_diff'] = (df['mort_diff'] - mean_diff) / std_diff
                df['z_product'] = df['z_betweenness'] * df['z_mort_diff']
                
                # 95th Percentile Rule
                thresh_val = df['z_product'].quantile(0.95)
                bridges = df[df['z_product'] >= thresh_val].copy()
                bridges['Sex'] = sex
                bridges['Age_Group'] = age_group
                return bridges
                
    return pd.DataFrame()


# ==============================================================================
# EDGE WEIGHT EXTRACTION (For Distribution Plots)
# ==============================================================================

def extract_edge_weights(
    outliers_df: pd.DataFrame, 
    sinks_df: pd.DataFrame, 
    bridges_df: pd.DataFrame,
    icd_df: pd.DataFrame
) -> Dict[str, List[float]]:
    """
    Extracts weights exactly as done in Notebook Cell 22.
    
    CRITICAL: The notebook iterates through the *identified nodes* and pulls 
    connections. Because matrices are symmetric, this often results in 
    finding edge A-B when processing A, and edge B-A when processing B. 
    We preserve this behavior to match the notebook's histograms (N~31k, N~11k).
    """
    logger.info("Extracting edge weights (replicating notebook logic)...")
    
    weights_data = {'Outliers': [], 'Sinks': [], 'Bridges': []}
    icd_to_node = dict(zip(icd_df['icd_code'], icd_df['diagnose_id'] - 1))
    
    # 1. Process Outliers & Sinks (Node-based iteration)
    # We define a helper to match `get_edge_weights_for_nodes` from Cell 22
    def get_weights_for_codes(codes_list, sex, age):
        w_list = []
        A = load_adjacency_matrix(sex, age) # Loads weighted matrix
        if A is None: return []
        
        target_nodes = [icd_to_node.get(c) for c in codes_list if c in icd_to_node]
        target_nodes = [n for n in target_nodes if n is not None]
        
        for node in target_nodes:
            # Notebook logic: iterate all neighbors.
            # "Outgoing" (Row)
            for neighbor in range(len(A)):
                if neighbor != node and A[node, neighbor] >= 1.5:
                    w_list.append(A[node, neighbor])
            
            # "Incoming" (Col) - In undirected, this doubles the count if symmetric
            # The notebook includes this block explicitly.
            for neighbor in range(len(A)):
                if neighbor != node and A[neighbor, node] >= 1.5:
                    w_list.append(A[neighbor, node])
        return w_list

    for category, df_source in [('Outliers', outliers_df), ('Sinks', sinks_df)]:
        if df_source.empty: continue
        
        # Group by Sex/Age to load matrix once per group
        for (sex, age), group in tqdm(df_source.groupby(['Sex', 'Age_Group']), desc=f"Weights: {category}"):
            codes = group['icd_code'].unique()
            w = get_weights_for_codes(codes, sex, age)
            weights_data[category].extend(w)

    # 2. Process Bridges (Edge-based iteration)
    if not bridges_df.empty:
        for (sex, age), group in tqdm(bridges_df.groupby(['Sex', 'Age_Group']), desc="Weights: Bridges"):
            A = load_adjacency_matrix(sex, age)
            if A is None: continue
            
            for _, row in group.iterrows():
                u = icd_to_node.get(row['icd1'])
                v = icd_to_node.get(row['icd2'])
                if u is not None and v is not None:
                    # Check weight in matrix
                    w = A[u, v]
                    if w >= 1.5:
                        weights_data['Bridges'].append(w)
                        
    return weights_data


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_edge_distributions(weights_data: Dict[str, List[float]]):
    """Reproduce Figure 1, 2, 3 from Notebook Cell 22."""
    
    # Set style locally to ensure match
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl") 
    
    # --- Figure 1: Separate Histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['Outliers', 'Sinks', 'Bridges']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        w = weights_data[metric]
        color = COLORS[metric]
        
        if w:
            # Log bins
            log_bins = np.logspace(np.log10(1.5), np.log10(max(w)), 50)
            ax.hist(w, bins=log_bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xscale('log')
            ax.set_xlim(1.5, max(w) * 1.1)
            
            # Lines
            ax.axvline(np.mean(w), color='red', linestyle='--', label=f'Mean: {np.mean(w):.2f}')
            ax.axvline(np.median(w), color='orange', linestyle='--', label=f'Median: {np.median(w):.2f}')
            
            ax.set_title(f'{metric}\n(n={len(w)} edges)', fontweight='bold')
            ax.set_xlabel('Odds Ratio (log scale)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.legend()
            
            # Stats Box
            stats_text = f"Min: {np.min(w):.2f}\nMax: {np.max(w):.2f}\nStd: {np.std(w):.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    plt.suptitle('Distribution of Edge Weights (Odds Ratios > 1.5)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_FIG_DIR / 'edge_weight_histograms_separate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 2: Overlaid ---
    plt.figure(figsize=(10, 6))
    all_w = []
    for m in metrics: all_w.extend(weights_data[m])
    log_bins = np.logspace(np.log10(1.5), np.log10(max(all_w)), 50) if all_w else 50
    
    for metric in metrics:
        w = weights_data[metric]
        if w:
            plt.hist(w, bins=log_bins, color=COLORS[metric], alpha=0.5, 
                     label=f'{metric} (n={len(w)})', edgecolor='black', linewidth=0.5)
    
    plt.xscale('log')
    plt.xlabel('Odds Ratio (log scale)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Comparison of Edge Weight Distributions (OR > 1.5)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_FIG_DIR / 'edge_weight_histograms_overlaid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 3: Boxplots ---
    plt.figure(figsize=(8, 6))
    data = [weights_data[m] for m in metrics if weights_data[m]]
    labels = [f"{m}\n(n={len(weights_data[m])})" for m in metrics if weights_data[m]]
    colors = [COLORS[m] for m in metrics if weights_data[m]]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, 
                     medianprops=dict(color='red', linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.yscale('log')
    plt.ylabel('Odds Ratio (log scale)', fontweight='bold')
    plt.title('Distribution Comparison of Edge Weights (OR > 1.5)', fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_FIG_DIR / 'edge_weight_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_comparison(
    summary_df: pd.DataFrame, 
    jaccard_scores: Dict[str, float]
):
    """Reproduce Threshold Bars and Jaccard Plot."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # --- Figure 4: Threshold Bars ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['Outliers', 'Sinks', 'Bridges']
    colors_list = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # Bar Plot
        x = range(len(summary_df))
        y = summary_df[metric]
        bars = ax.bar(x, y, color=colors_list[idx], alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['OR > 1.5\n(Baseline)', 'OR > 2.0\n(Strict)'])
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(int(val)), ha='center', va='bottom', fontweight='bold')
            
    plt.suptitle('Counts of Critical Nodes/Edges Across OR Thresholds', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_FIG_DIR / 'threshold_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 5: Jaccard ---
    plt.figure(figsize=(8, 6))
    names = ['Outliers', 'Sinks', 'Bridges']
    vals = [jaccard_scores[n] for n in names]
    
    bars = plt.bar(range(3), vals, color=colors_list, alpha=0.8, edgecolor='black')
    
    plt.xticks(range(3), names)
    plt.ylabel('Jaccard Index', fontweight='bold')
    plt.title('Robustness: OR > 1.5 vs OR > 2.0', fontweight='bold')
    plt.ylim(0, 1.05)
    
    # Threshold lines
    plt.axhline(0.7, color='green', linestyle='--', label='J > 0.70')
    plt.axhline(0.4, color='orange', linestyle='--', label='J > 0.40')
    plt.legend(loc='lower left')
    
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_FIG_DIR / 'threshold_jaccard_indices.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

@app.command()
def main():
    logger.info("Starting COMPLETE THRESHOLD ROBUSTNESS ANALYSIS (Notebook Replication)")
    
    # 1. Load Reference Data
    icd_df = load_icd_mapping()
    prev_df = load_prevalence()
    mort_df = load_mortality()
    
    results = {}
    
    # 2. Iterate Thresholds
    for key, cfg in THRESHOLDS.items():
        t_name = cfg['name']
        t_val = cfg['value']
        logger.info(f"Processing {t_name}...")
        
        outliers_list = []
        sinks_list = []
        bridges_list = []
        
        for sex in SEXES:
            for age_id, age_str in AGE_GROUPS.items():
                logger.debug(f"  > Processing {sex} {age_str}")
                
                # Filter Prevalence/Mortality for this group
                prev_sub = prev_df[
                    (prev_df['sex'] == sex) & 
                    (prev_df['Age_Group'] == age_str) & 
                    (prev_df['year'] == 2014)
                ]
                prev_dict = dict(zip(prev_sub['icd_code'], prev_sub['p']))
                
                mort_sub = mort_df[
                    (mort_df['sex'] == sex) & 
                    (mort_df['age_10'] == age_id) # note: mort df uses int age_10 usually
                ]
                mort_dict = dict(zip(mort_sub['icd_code'], mort_sub['mortality']))
                
                # -- Computation --
                try:
                    out = compute_degree_outliers(sex, age_id, t_val, icd_df, prev_dict)
                    snk = compute_high_mortality_sinks(sex, age_id, t_val, icd_df, mort_dict)
                    brg = compute_high_mortality_bridges(sex, age_id, t_val, icd_df, mort_dict)
                    
                    if not out.empty: outliers_list.append(out)
                    if not snk.empty: sinks_list.append(snk)
                    if not brg.empty: bridges_list.append(brg)
                except Exception as e:
                    logger.error(f"Error in {sex} {age_str}: {e}")
                    
        # Consolidate
        df_out = pd.concat(outliers_list) if outliers_list else pd.DataFrame()
        df_snk = pd.concat(sinks_list) if sinks_list else pd.DataFrame()
        df_brg = pd.concat(bridges_list) if bridges_list else pd.DataFrame()
        
        # Create Sets for Jaccard
        set_out = set(df_out.apply(lambda r: f"{r['Sex']}_{r['Age_Group']}_{r['icd_code']}", axis=1)) if not df_out.empty else set()
        set_snk = set(df_snk.apply(lambda r: f"{r['Sex']}_{r['Age_Group']}_{r['icd_code']}", axis=1)) if not df_snk.empty else set()
        set_brg = set(df_brg.apply(lambda r: f"{r['Sex']}_{r['Age_Group']}_{r['icd1']}_{r['icd2']}", axis=1)) if not df_brg.empty else set()
        
        results[key] = {
            'dfs': (df_out, df_snk, df_brg),
            'sets': (set_out, set_snk, set_brg),
            'counts': (len(set_out), len(set_snk), len(set_brg))
        }
        
        logger.info(f"  Results for {t_name}: Outliers={len(set_out)}, Sinks={len(set_snk)}, Bridges={len(set_brg)}")

    # 3. Compute Jaccard
    def get_jaccard(s1, s2):
        u = len(s1 | s2)
        return len(s1 & s2) / u if u > 0 else 0
        
    jaccards = {
        'Outliers': get_jaccard(results['or_1.5']['sets'][0], results['or_2.0']['sets'][0]),
        'Sinks': get_jaccard(results['or_1.5']['sets'][1], results['or_2.0']['sets'][1]),
        'Bridges': get_jaccard(results['or_1.5']['sets'][2], results['or_2.0']['sets'][2]),
    }
    
    # 4. Save Summary CSV
    summary_data = [
        {'Condition': 'OR > 1.5', 'Outliers': results['or_1.5']['counts'][0], 'Sinks': results['or_1.5']['counts'][1], 'Bridges': results['or_1.5']['counts'][2]},
        {'Condition': 'OR > 2.0', 'Outliers': results['or_2.0']['counts'][0], 'Sinks': results['or_2.0']['counts'][1], 'Bridges': results['or_2.0']['counts'][2]}
    ]
    pd.DataFrame(summary_data).to_csv(PROCESSED_DATA_DIR / 'threshold_robustness_summary.csv', index=False)
    
    # 5. Extract Weights (Baseline Only)
    logger.info("Analyzing Edge Weight Distributions for OR > 1.5...")
    base_dfs = results['or_1.5']['dfs'] # (Out, Snk, Brg)
    weights_map = extract_edge_weights(base_dfs[0], base_dfs[1], base_dfs[2], icd_df)
    
    # 6. Generate All Plots
    logger.info("Generating Plots...")
    plot_threshold_comparison(pd.DataFrame(summary_data), jaccards)
    plot_edge_distributions(weights_map)
    
    logger.success(f"Analysis Complete. Figures saved to {ROBUSTNESS_FIG_DIR}")

if __name__ == "__main__":
    app()