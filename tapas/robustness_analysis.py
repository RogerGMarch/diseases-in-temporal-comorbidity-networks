"""
Robustness analysis module.
Implements the sensitivity analysis (OR > 1.5 vs OR > 2.0) and edge weight distributions
from the RobustnessTest notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from pathlib import Path
from loguru import logger
from typing import Set, Dict, List, Tuple

from tapas.config import PROCESSED_DATA_DIR, FIGURES_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

# Import core calculation logic from the analysis modules
from tapas.analysis.outliers import detect_outliers_exact, identify_high_mortality_sinks_zscore
from tapas.analysis.bridges import identify_critical_bridges

app = typer.Typer()

THRESHOLDS = {
    'or_1.5': {'name': 'OR > 1.5 (Baseline)', 'value': 1.5},
    'or_2.0': {'name': 'OR > 2.0 (Strict)', 'value': 2.0}
}

def create_unique_identifier(row, is_edge=False) -> str:
    """Create unique identifier for node or edge to compare sets."""
    if is_edge:
        # Sort codes to ensure undirected edges are consistent
        c1, c2 = sorted([row['ICD_Code_1'], row['ICD_Code_2']])
        return f"{row['Sex']}_{row['Age_Group']}_{c1}_{c2}"
    else:
        return f"{row['Sex']}_{row['Age_Group']}_{row['ICD_Code']}"

def compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard index between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return np.nan
    union = set1 | set2
    if len(union) == 0:
        return 0.0
    intersection = set1 & set2
    return len(intersection) / len(union)

def get_raw_edge_weights(icd_codes: List[str], sex: str, age_group: int) -> List[float]:
    """
    Extract raw edge weights for specific nodes from the adjacency matrix.
    Returns unthresholded edge weights to examine distribution of connection strengths.
    """
    paths = NetworkAnalyzer._resolve_paths(sex, age_group)
    if not paths["adjacency"].exists():
        return []
    
    # Load RAW matrix
    try:
        adj = np.loadtxt(paths["adjacency"], delimiter=" ")
    except Exception:
        return []

    # Map ICD to index (0-based) using the standard ICD file
    try:
        icd_df = pd.read_csv(paths["icd_codes"])
        # Handle column name mapping: Id->diagnose_id, Code->icd_code, ShortDescription->descr
        if 'Id' in icd_df.columns:
            icd_df = icd_df.rename(columns={'Id': 'diagnose_id', 'Code': 'icd_code', 'ShortDescription': 'descr'})
        # Map: ICD code -> diagnose_id (1-based) -> index (0-based)
        icd_to_idx = dict(zip(icd_df['icd_code'], icd_df['diagnose_id'] - 1))
    except Exception:
        return []

    weights = []
    target_indices = {icd_to_idx[code] for code in icd_codes if code in icd_to_idx}
    
    num_nodes = len(adj)
    
    for u in target_indices:
        # Check all neighbors
        # Optimized: only check rows where u is involved
        # Adjacency matrix is symmetric for our usage (or assumed so)
        if u >= num_nodes: continue
        
        row = adj[u]
        # Filter: neighbors with OR >= 1.5 (baseline threshold)
        neighbors = np.where(row >= 1.5)[0]
        
        for v in neighbors:
            if u != v: # No self-loops usually
                weights.append(row[v])
                
    return weights

def get_bridge_weights(bridges_df: pd.DataFrame) -> List[float]:
    """Extract specific weights for identified bridge edges."""
    weights = []
    # Group by sex/age to minimize file I/O
    for (sex, age), group in bridges_df.groupby(['Sex', 'Age_Group']):
        paths = NetworkAnalyzer._resolve_paths(sex, age)
        if not paths["adjacency"].exists(): continue
        try:
            adj = np.loadtxt(paths["adjacency"], delimiter=" ")
            icd_df = pd.read_csv(paths["icd_codes"])
            # Handle column name mapping: Id->diagnose_id, Code->icd_code, ShortDescription->descr
            if 'Id' in icd_df.columns:
                icd_df = icd_df.rename(columns={'Id': 'diagnose_id', 'Code': 'icd_code', 'ShortDescription': 'descr'})
            icd_to_idx = dict(zip(icd_df['icd_code'], icd_df['diagnose_id'] - 1))
        except: continue
        
        for _, row in group.iterrows():
            u_code, v_code = row['ICD_Code_1'], row['ICD_Code_2']
            if u_code in icd_to_idx and v_code in icd_to_idx:
                u, v = icd_to_idx[u_code], icd_to_idx[v_code]
                if u < len(adj) and v < len(adj):
                    w = adj[u, v]
                    if w >= 1.5: weights.append(w)
    return weights

@app.command()
def complete_analysis():
    """
    Run complete threshold robustness analysis (OR 1.5 vs 2.0).
    Generates comparison stats, Jaccard indices, and edge weight histograms.
    """
    logger.info("Starting Complete Robustness Analysis...")
    output_dir = PROCESSED_DATA_DIR / "robustness"
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = FIGURES_DIR / "robustness"
    fig_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    # ---------------------------------------------------------
    # PART 1: Threshold Comparison (Metrics & Identification)
    # ---------------------------------------------------------
    for key, info in THRESHOLDS.items():
        threshold = info['value']
        logger.info(f"Processing Threshold: {info['name']} (value={threshold})")
        
        # 1. Load Data with Threshold
        node_dfs = []
        edge_dfs = []
        
        for sex in SEXES:
            for age in AGE_GROUPS.keys():
                # Load with specific threshold
                ndf = NetworkAnalyzer.load_node_metrics(sex, age, threshold=threshold)
                if not ndf.empty and 'Prevalence' in ndf.columns:
                    # Filter prevalence > 0 (consistency with outlier logic)
                    ndf = ndf[ndf['Prevalence'] > 0]
                    node_dfs.append(ndf)
                
                edf = NetworkAnalyzer.load_edge_metrics(sex, age, threshold=threshold)
                if not edf.empty:
                    edge_dfs.append(edf)
        
        if not node_dfs: continue
        df_nodes = pd.concat(node_dfs, ignore_index=True)
        df_edges = pd.concat(edge_dfs, ignore_index=True)
        
        # 2. Identify Critical Sets
        # Outliers (Degree)
        outliers = detect_outliers_exact(df_nodes)
        # Filter actual outliers
        outliers = outliers[outliers['Outlier'] == True].copy()
        
        # Sinks (Mortality)
        sinks = identify_high_mortality_sinks_zscore(df_nodes, top_percent=20)
        
        # Bridges (Edges)
        bridges = identify_critical_bridges(df_edges, top_percent=5, min_mort_diff=0.30) # Default params
        
        # 3. Store Results
        results[key] = {
            'outliers': outliers,
            'sinks': sinks,
            'bridges': bridges,
            'outliers_set': set(outliers.apply(lambda r: create_unique_identifier(r), axis=1)),
            'sinks_set': set(sinks.apply(lambda r: create_unique_identifier(r), axis=1)),
            'bridges_set': set(bridges.apply(lambda r: create_unique_identifier(r, True), axis=1))
        }
        
        # Save CSVs
        outliers.to_csv(output_dir / f"threshold_{key}_outliers.csv", index=False)
        sinks.to_csv(output_dir / f"threshold_{key}_sinks.csv", index=False)
        bridges.to_csv(output_dir / f"threshold_{key}_bridges.csv", index=False)
        logger.info(f"  Saved CSVs for {key}")

    # ---------------------------------------------------------
    # PART 2: Overlap Statistics (Jaccard)
    # ---------------------------------------------------------
    if 'or_1.5' in results and 'or_2.0' in results:
        r1, r2 = results['or_1.5'], results['or_2.0']
        
        stats = {
            'comparison': 'OR > 1.5 vs OR > 2.0',
            'outliers_jaccard': compute_jaccard(r1['outliers_set'], r2['outliers_set']),
            'sinks_jaccard': compute_jaccard(r1['sinks_set'], r2['sinks_set']),
            'bridges_jaccard': compute_jaccard(r1['bridges_set'], r2['bridges_set']),
            'outliers_count_1.5': len(r1['outliers_set']),
            'outliers_count_2.0': len(r2['outliers_set']),
            # Add more counts as needed
        }
        
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_dir / "overlap_statistics.csv", index=False)
        
        # Plot Jaccard
        plt.figure(figsize=(8, 6))
        metrics = ['Outliers', 'Sinks', 'Bridges']
        values = [stats['outliers_jaccard'], stats['sinks_jaccard'], stats['bridges_jaccard']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = plt.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
        plt.ylim(0, 1.1)
        plt.axhline(0.7, color='green', linestyle='--', label='J > 0.70')
        plt.title("Robustness: OR > 1.5 vs OR > 2.0", fontsize=14, fontweight='bold')
        plt.ylabel("Jaccard Index")
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')
            
        plt.legend()
        plt.savefig(fig_dir / "threshold_jaccard_indices.png", dpi=300)
        plt.close()
        logger.success("Generated Jaccard plot.")

    # ---------------------------------------------------------
    # PART 3: Edge Weight Distributions (Baseline 1.5)
    # ---------------------------------------------------------
    if 'or_1.5' in results:
        logger.info("Analyzing Edge Weight Distributions (Baseline 1.5)...")
        r_base = results['or_1.5']
        
        # Collect weights
        # Outliers & Sinks (Need to look up weights for collected nodes)
        outlier_nodes = r_base['outliers'][['ICD_Code', 'Sex', 'Age_Group']].drop_duplicates()
        sink_nodes = r_base['sinks'][['ICD_Code', 'Sex', 'Age_Group']].drop_duplicates()
        
        outlier_weights = []
        for (sex, age), group in outlier_nodes.groupby(['Sex', 'Age_Group']):
            w = get_raw_edge_weights(group['ICD_Code'].tolist(), sex, int(age))
            outlier_weights.extend(w)
            
        sink_weights = []
        for (sex, age), group in sink_nodes.groupby(['Sex', 'Age_Group']):
            w = get_raw_edge_weights(group['ICD_Code'].tolist(), sex, int(age))
            sink_weights.extend(w)
            
        # Bridges (Direct lookup)
        bridge_weights = get_bridge_weights(r_base['bridges'])
        
        all_weights = outlier_weights + sink_weights + bridge_weights
        
        if all_weights:
            max_val = max(all_weights)
            
            # --- PLOT 1: Separate Histograms (Side-by-Side) ---
            # Using log bins common across all for comparison.
            log_bins = np.logspace(np.log10(1.5), np.log10(max_val), 50)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            categories = [
                ('Outliers', outlier_weights, '#3498db'),
                ('Sinks', sink_weights, '#e74c3c'),
                ('Bridges', bridge_weights, '#2ecc71')
            ]
            
            for idx, (name, weights, color) in enumerate(categories):
                ax = axes[idx]
                if weights:
                    ax.hist(weights, bins=log_bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.set_xscale('log')
                    ax.set_xlim(1.5, max_val * 1.1)
                    
                    # Statistics lines
                    mean_w = np.mean(weights)
                    median_w = np.median(weights)
                    ax.axvline(mean_w, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_w:.2f}')
                    ax.axvline(median_w, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_w:.2f}')
                    
                    ax.set_xlabel('Odds Ratio (log scale)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    ax.set_title(f'{name}\n(n={len(weights)} edges)', fontsize=13, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=9)
                    ax.grid(axis='both', alpha=0.3, linestyle='--')
                    
                    # Add statistics text box
                    stats_text = f'Min: {np.min(weights):.2f}\nMax: {np.max(weights):.2f}\nStd: {np.std(weights):.2f}'
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

            plt.suptitle('Distribution of Edge Weights (Odds Ratios > 1.5)', fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(fig_dir / "edge_weight_histograms_separate.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.success("Generated Separate Edge Weight Histograms.")

            # --- PLOT 2: Overlaid Histogram ---
            plt.figure(figsize=(10, 6))
            plt.hist(outlier_weights, bins=log_bins, color='#3498db', alpha=0.5, label=f'Outliers (n={len(outlier_weights)})')
            plt.hist(sink_weights, bins=log_bins, color='#e74c3c', alpha=0.5, label=f'Sinks (n={len(sink_weights)})')
            plt.hist(bridge_weights, bins=log_bins, color='#2ecc71', alpha=0.5, label=f'Bridges (n={len(bridge_weights)})')
            
            plt.xscale('log')
            plt.xlabel("Odds Ratio (log scale)", fontweight='bold')
            plt.ylabel("Frequency", fontweight='bold')
            plt.title("Comparison of Edge Weight Distributions (OR > 1.5)", fontweight='bold')
            plt.legend()
            
            plt.savefig(fig_dir / "edge_weight_histograms_overlaid.png", dpi=300)
            plt.close()
            logger.success("Generated Overlaid Edge Weight Histogram.")
            
            # --- PLOT 3: Box Plots ---
            plt.figure(figsize=(8, 6))
            data_to_plot = []
            labels = []
            colors_bp = []
            
            for name, w, c in categories:
                if w:
                    data_to_plot.append(w)
                    labels.append(f'{name}\n(n={len(w)})')
                    colors_bp.append(c)
            
            bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                             medianprops=dict(color='red', linewidth=2),
                             boxprops=dict(facecolor='lightblue', alpha=0.7))
            
            for patch, color in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            plt.yscale('log')
            plt.ylabel('Odds Ratio (log scale)', fontsize=12, fontweight='bold')
            plt.title('Distribution Comparison (Boxplots)', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.savefig(fig_dir / "edge_weight_boxplots.png", dpi=300)
            plt.close()
            logger.success("Generated Boxplots.")

            # Save Statistics
            stats = []
            for name, w in [('Outliers', outlier_weights), ('Sinks', sink_weights), ('Bridges', bridge_weights)]:
                if w:
                    stats.append({
                        'Category': name,
                        'Count': len(w),
                        'Mean': np.mean(w),
                        'Median': np.median(w),
                        'Max': np.max(w),
                        'Min': np.min(w),
                        'Std': np.std(w)
                    })
            pd.DataFrame(stats).to_csv(output_dir / "edge_weight_statistics.csv", index=False)

if __name__ == "__main__":
    app()