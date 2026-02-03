import pandas as pd
import numpy as np
from pathlib import Path
import typer
from loguru import logger
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import project configuration and features
from tapas.config import (
    DATA_DIR, 
    PROCESSED_DATA_DIR, 
    INTERIM_DATA_DIR,
    AGE_GROUPS, 
    SEXES,
    REPORTS_DIR,
    FIGURES_DIR
)
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def load_network_and_prevalence(gender: str, age_group_id: int) -> pd.DataFrame:
    """
    Load network data using NetworkAnalyzer and calculate degree/prevalence.
    
    Args:
        gender: 'Male' or 'Female'
        age_group_id: Integer 1-8
    """
    age_label = AGE_GROUPS[age_group_id]
    
    # 1. Define Paths (Adjusting based on structure implied in features.py comments)
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data" 
    
    # If the specific folder structure differs, fallback to DATA_DIR
    if not base_data_path.exists():
        base_data_path = DATA_DIR

    adj_path = base_data_path / "3.AdjacencyMatrices" / f"Adj_Matrix_{gender}_ICD_age_{age_group_id}.csv"
    icd_path = base_data_path / "ICD10_Diagnoses_All.csv"
    prev_path = base_data_path / "1.Prevalence" / "Prevalence_Sex_Age_Year_ICD.csv"
    
    if not adj_path.exists():
        logger.error(f"Adjacency file not found: {adj_path}")
        return pd.DataFrame()

    # 2. Load Graph using functionality from features.py
    try:
        G = NetworkAnalyzer.load_adjacency_matrix(adj_path)
        analyzer = NetworkAnalyzer(G)
        # We need raw degrees for the outlier calculation
        graph_obj = analyzer.graph
    except Exception as e:
        logger.error(f"Failed to load graph for {gender} {age_label}: {e}")
        return pd.DataFrame()

    # 3. Load Metadata (ICD and Prevalence)
    try:
        icd_df = pd.read_csv(icd_path)
        prev_df = pd.read_csv(prev_path)
    except FileNotFoundError as e:
        logger.error(f"Metadata file missing: {e}")
        return pd.DataFrame()

    # 4. Filter Prevalence for 2014
    prev_2014 = prev_df[
        (prev_df['Age_Group'] == age_label) & 
        (prev_df['sex'] == gender) & 
        (prev_df['year'] == 2014)
    ]
    prevalence_dict = prev_2014.set_index('icd_code')['p'].to_dict()

    # 5. Build Results
    results = []
    
    for node_idx in graph_obj.nodes():
        degree = graph_obj.degree(node_idx)
        
        if degree > 0:
            diagnose_id = node_idx + 1
            icd_row = icd_df[icd_df['diagnose_id'] == diagnose_id]
            
            if len(icd_row) > 0:
                icd_code = icd_row.iloc[0]['icd_code']
                descr = icd_row.iloc[0]['descr']
                prevalence = prevalence_dict.get(icd_code, 0)
                
                if prevalence > 0:
                    results.append({
                        'Sex': gender,
                        'Age_Group': f'age_{age_group_id}',
                        'Age_Range': age_label,
                        'ICD_Code': icd_code,
                        'Degree': degree,
                        'Prevalence': prevalence,
                        'Description_GER': descr
                    })
    
    return pd.DataFrame(results)

def modified_zscore(x, median, mad):
    """Modified z-score: 0.6745 * (x - median) / mad"""
    if mad == 0:
        return np.nan
    return 0.6745 * (x - median) / mad

def detect_outliers_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the exact method from the notebook (Cell 20).
    Uses 20th/80th percentiles and Modified Z-Score.
    """
    logger.info("Detecting outliers using Exact Notebook Method (20th/80th percentile)...")
    
    # Calculate Ratio and Log Ratio
    df['Ratio'] = df['Degree'] / df['Prevalence']
    df['Log_ratio'] = df['Ratio'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    df_out = pd.DataFrame()
    
    # Process each sex-age group separately
    for sex in df['Sex'].unique():
        for age_id in df['Age_Group'].unique():
            df_subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_id)].copy()
            
            if len(df_subset) == 0:
                continue
            
            # Calculate percentiles (exact method from notebook)
            lower_bound = df_subset['Log_ratio'].quantile(0.2)
            upper_bound = df_subset['Log_ratio'].quantile(0.80)
            
            # Modified z-score
            median = df_subset['Log_ratio'].median()
            mad = (df_subset['Log_ratio'] - median).abs().median()
            df_subset['Deviation'] = df_subset['Log_ratio'].apply(
                lambda x: modified_zscore(x, median, mad)
            )
            
            # Mark outliers
            df_subset['Outlier'] = (
                (df_subset['Log_ratio'] < lower_bound) | 
                (df_subset['Log_ratio'] > upper_bound)
            )
            
            df_out = pd.concat([df_out, df_subset], ignore_index=True)
            
    return df_out

def detect_outliers_logic(df_all: pd.DataFrame) -> pd.DataFrame:
    """Apply 20th/80th percentile thresholding on Log Ratio (Legacy/Simpler version)."""
    # This logic is effectively superseded by detect_outliers_exact but kept for compatibility
    return detect_outliers_exact(df_all)

def add_english_descriptions(df_outliers: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """Add English descriptions mapping."""
    logger.info("Adding English descriptions...")
    
    eng_path = base_path / 'ICD10_Diagnoses_All_ENG.csv'
    
    if not eng_path.exists():
        logger.warning(f"English description file not found at {eng_path}. Skipping translation.")
        if 'Description_Eng' not in df_outliers.columns:
            df_outliers['Description_Eng'] = df_outliers['Description_GER']
        return df_outliers

    eng_df = pd.read_csv(eng_path)
    if 'Code' in eng_df.columns and 'ShortDescription' in eng_df.columns:
        icd_to_eng = dict(zip(eng_df['Code'], eng_df['ShortDescription']))
        df_outliers['Description_Eng'] = df_outliers['ICD_Code'].map(icd_to_eng)
        df_outliers['Description_Eng'] = df_outliers['Description_Eng'].fillna(df_outliers['Description_GER'])
    else:
        logger.warning("English dictionary columns format not recognized.")
        if 'Description_Eng' not in df_outliers.columns:
            df_outliers['Description_Eng'] = df_outliers['Description_GER']
        
    return df_outliers

def select_top_outliers(df_outliers: pd.DataFrame, n_high: int = 20, n_low: int = 10) -> pd.DataFrame:
    """Select top N high and low degree outliers per sex-age group"""
    logger.info(f"Selecting top {n_high} high and top {n_low} low outliers per group...")
    
    results = []
    
    for sex in ['Female', 'Male']:
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        
        for age_range in sorted(sex_data['Age_Range'].unique()):
            age_data = sex_data[sex_data['Age_Range'] == age_range]
            
            if len(age_data) == 0:
                continue
            
            age_median = age_data['Log_ratio'].median()
            
            high_degree = age_data[age_data['Log_ratio'] > age_median].nlargest(n_high, 'Log_ratio')
            high_degree = high_degree.copy()
            high_degree['outlier_type'] = 'high_degree'
            
            low_degree = age_data[age_data['Log_ratio'] <= age_median].nsmallest(n_low, 'Log_ratio')
            low_degree = low_degree.copy()
            low_degree['outlier_type'] = 'low_degree'
            
            results.append(high_degree)
            results.append(low_degree)
    
    table_data = pd.concat(results, ignore_index=True)
    
    table_data['age_num'] = table_data['Age_Range'].map({
        '0-9': 1, '10-19': 2, '20-29': 3, '30-39': 4,
        '40-49': 5, '50-59': 6, '60-69': 7, '70-79': 8
    })
    table_data['type_order'] = table_data['outlier_type'].map({
        'high_degree': 0, 'low_degree': 1
    })
    
    table_data = table_data.sort_values(
        ['Sex', 'age_num', 'type_order', 'Log_ratio'], 
        ascending=[True, True, True, False]
    )
    
    return table_data

# ========================================================================================
# Mortality Sinks Functions
# ========================================================================================

def load_network_and_mortality(gender: str, age_group_id: int) -> pd.DataFrame:
    """
    Load network data and mortality data to calculate betweenness and mortality.
    """
    age_label = AGE_GROUPS[age_group_id]
    
    # Define Paths
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR

    adj_path = base_data_path / "3.AdjacencyMatrices" / f"Adj_Matrix_{gender}_ICD_age_{age_group_id}.csv"
    icd_path = base_data_path / "ICD10_Diagnoses_All.csv"
    
    # Mortality file path handling
    mort_file = f"mortality_diag_{gender}.csv"
    mort_path = base_data_path / mort_file
    if not mort_path.exists():
        mort_path = DATA_DIR / mort_file
    
    if not adj_path.exists() or not mort_path.exists():
        logger.warning(f"Missing files for {gender} {age_label}: Adj={adj_path.exists()}, Mort={mort_path.exists()}")
        return pd.DataFrame()

    # Load Graph
    try:
        G = NetworkAnalyzer.load_adjacency_matrix(adj_path)
        analyzer = NetworkAnalyzer(G)
        # Calculate betweenness using normalized=True
        betweenness = analyzer.get_node_betweenness(normalized=True)
        graph_obj = analyzer.graph
    except Exception as e:
        logger.error(f"Failed to load graph for {gender} {age_label}: {e}")
        return pd.DataFrame()

    # Load Metadata
    try:
        icd_df = pd.read_csv(icd_path)
        mortality_df = pd.read_csv(mort_path)
    except Exception as e:
        logger.error(f"Metadata file error: {e}")
        return pd.DataFrame()

    # Filter Mortality for this age group (using age_10 column from notebook)
    if 'age_10' not in mortality_df.columns:
        logger.error(f"Mortality file missing 'age_10' column")
        return pd.DataFrame()
        
    mortality_age = mortality_df[mortality_df['age_10'] == age_group_id]
    mortality_dict = dict(zip(mortality_age['icd_code'], mortality_age['mortality']))

    results = []
    for node_idx in graph_obj.nodes():
        degree = graph_obj.degree(node_idx)
        if degree > 0:
            diagnose_id = node_idx + 1
            icd_row = icd_df[icd_df['diagnose_id'] == diagnose_id]
            
            if len(icd_row) > 0:
                icd_code = icd_row.iloc[0]['icd_code']
                descr = icd_row.iloc[0]['descr']
                
                bet = betweenness.get(node_idx, 0)
                mort = mortality_dict.get(icd_code, 0)
                
                results.append({
                    'Sex': gender,
                    'Age_Group': age_group_id, # Keep as int for processing logic
                    'Age_Range': age_label,
                    'ICD_Code': icd_code,
                    'Description_GER': descr,
                    'Degree': degree,
                    'Betweenness': bet,
                    'Mortality': mort
                })
    return pd.DataFrame(results)

def identify_high_mortality_sinks_zscore(df_all: pd.DataFrame, top_percent: int = 20) -> pd.DataFrame:
    """
    Identify high-mortality sinks using Z-score product method.
    """
    logger.info(f"Identifying high-mortality sinks (Top {top_percent}% Z-score product)...")
    all_high = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group)
            ].copy()
            
            if len(subset) == 0: continue
            
            # Calculate Z-scores
            for col in ['Betweenness', 'Mortality']:
                mean = subset[col].mean()
                std = subset[col].std()
                if std > 0:
                    subset[f'z_{col.lower()}'] = (subset[col] - mean) / std
                else:
                    subset[f'z_{col.lower()}'] = 0
            
            # Calculate Product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mortality']
            
            # Calculate geometric mean (for ranking/display, handling only positive Zs)
            subset['z_geom_mean'] = np.where(
                (subset['z_betweenness'] > 0) & (subset['z_mortality'] > 0),
                np.sqrt(subset['z_betweenness'] * subset['z_mortality']),
                0
            )
            
            # Thresholding
            threshold_percentile = 100 - top_percent
            z_threshold = subset['z_product'].quantile(threshold_percentile / 100)
            
            # Filter: must have positive Z-scores and product > threshold
            high_nodes = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mortality'] > 0) &
                (subset['z_product'] >= z_threshold)
            ].copy()
            
            if len(high_nodes) > 0:
                all_high.append(high_nodes)
                
    return pd.concat(all_high, ignore_index=True) if all_high else pd.DataFrame()


@app.command()
def mortality_sinks(
    output_filename: str = "high_mortality_sinks_ZSCORE.csv",
    top_percent: int = 20
):
    """
    Generate the High Mortality Sinks table (Z-score method).
    """
    logger.info("Starting Mortality Sinks Analysis (Z-score Method)...")
    
    all_data = []
    
    # Load Data
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age Group {age_id} ({AGE_GROUPS[age_id]})...")
            df = load_network_and_mortality(gender, age_id)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded. Check files.")
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    logger.success(f"Loaded {len(df_all)} nodes for analysis.")
    
    # Identify Sinks
    df_high = identify_high_mortality_sinks_zscore(df_all, top_percent=top_percent)
    
    if df_high.empty:
        logger.warning("No sinks identified.")
        return

    logger.success(f"Identified {len(df_high)} high-mortality sinks.")

    # Add English Descriptions
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR
    df_high = add_english_descriptions(df_high, base_data_path)
    
    # Save CSV
    out_csv = PROCESSED_DATA_DIR / output_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_high.to_csv(out_csv, index=False)
    logger.success(f"Saved CSV to {out_csv}")
    
    # Summary
    print("\n" + "="*60)
    print("MORTALITY SINKS SUMMARY")
    print("="*60)
    for sex in SEXES:
        sex_data = df_high[df_high['Sex'] == sex]
        print(f"\n{sex}: {len(sex_data)} nodes")
        for age_id in sorted(sex_data['Age_Group'].unique()):
            age_data = sex_data[sex_data['Age_Group'] == age_id]
            if len(age_data) > 0:
                age_lbl = AGE_GROUPS[age_id]
                print(f"  {age_lbl}: {len(age_data)} nodes")
                # Top 3
                top3 = age_data.nlargest(3, 'z_geom_mean')
                for _, row in top3.iterrows():
                     print(f"    - {row['ICD_Code']:4} (Z-GeoMean={row['z_geom_mean']:.3f}, Mort={row['Mortality']:.4f})")

# ========================================================================================
# Bridge Edges Functions
# ========================================================================================

def load_network_edges_with_mortality(gender: str, age_group_id: int) -> pd.DataFrame:
    """
    Load network edges with betweenness and mortality differences.
    """
    age_label = AGE_GROUPS[age_group_id]
    
    # Define Paths
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data" 
    if not base_data_path.exists():
        base_data_path = DATA_DIR

    adj_path = base_data_path / "3.AdjacencyMatrices" / f"Adj_Matrix_{gender}_ICD_age_{age_group_id}.csv"
    icd_path = base_data_path / "ICD10_Diagnoses_All.csv"
    
    mort_file = f"mortality_diag_{gender}.csv"
    mort_path = base_data_path / mort_file
    if not mort_path.exists():
        mort_path = DATA_DIR / mort_file
        
    if not adj_path.exists() or not mort_path.exists():
        logger.warning(f"Missing files for {gender} {age_label}")
        return pd.DataFrame()

    # Load Graph and Calculate Edge Betweenness
    try:
        G = NetworkAnalyzer.load_adjacency_matrix(adj_path)
        analyzer = NetworkAnalyzer(G)
        # Notebook uses unweighted edge betweenness (default weight=None)
        # NetworkAnalyzer.get_edge_betweenness uses normalized=True by default which matches nx default
        edge_betweenness = analyzer.get_edge_betweenness(normalized=True)
        graph_obj = analyzer.graph
    except Exception as e:
        logger.error(f"Graph error {gender} {age_label}: {e}")
        return pd.DataFrame()

    # Load Metadata
    try:
        icd_df = pd.read_csv(icd_path)
        # ICD Dict: diagnose_id in file is 1-based, 0-based in graph
        icd_dict = dict(zip(icd_df['diagnose_id'] - 1, icd_df['icd_code']))
        descr_dict = dict(zip(icd_df['diagnose_id'] - 1, icd_df['descr']))
        
        mortality_df = pd.read_csv(mort_path)
        mortality_age = mortality_df[mortality_df['age_10'] == age_group_id]
        mortality_dict = dict(zip(mortality_age['icd_code'], mortality_age['mortality']))
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        return pd.DataFrame()

    results = []
    # Loop over edges
    for u, v in graph_obj.edges():
        # Edge betweenness key is usually (u, v) or (v, u). 
        bet = edge_betweenness.get((u, v))
        if bet is None:
            bet = edge_betweenness.get((v, u), 0)
            
        icd1 = icd_dict.get(u)
        icd2 = icd_dict.get(v)
        
        if icd1 is None or icd2 is None:
            continue
            
        desc1 = descr_dict.get(u, '')
        desc2 = descr_dict.get(v, '')
        
        mort1 = mortality_dict.get(icd1, 0)
        mort2 = mortality_dict.get(icd2, 0)
        
        mort_diff = abs(mort1 - mort2)
        
        results.append({
            'Sex': gender,
            'Age_Group': age_group_id,
            'Age_Range': age_label,
            'ICD_Code_1': icd1,
            'ICD_Code_2': icd2,
            'Description_1': desc1,
            'Description_2': desc2,
            'Edge_Betweenness': bet,
            'Mortality_1': mort1,
            'Mortality_2': mort2,
            'Mortality_Diff': mort_diff
        })
        
    return pd.DataFrame(results)

def identify_critical_bridge_edges_zscore(df_all: pd.DataFrame, top_percent: float = 5, min_mort_diff: float = 0.30) -> pd.DataFrame:
    """
    Identify bridge edges using Z-score product method.
    """
    logger.info(f"Identifying bridge edges (Top {top_percent}% Z-score product, Min Diff > {min_mort_diff})...")
    all_bridges = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group)
            ].copy()
            
            if len(subset) == 0: continue
            
            # Z-scores
            for col in ['Edge_Betweenness', 'Mortality_Diff']:
                mean = subset[col].mean()
                std = subset[col].std()
                col_name = 'z_betweenness' if col == 'Edge_Betweenness' else 'z_mort_diff'
                if std > 0:
                    subset[col_name] = (subset[col] - mean) / std
                else:
                    subset[col_name] = 0
            
            # Z-Score Product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            
            threshold_percentile = 100 - top_percent
            z_threshold = subset['z_product'].quantile(threshold_percentile / 100)
            
            bridge_edges = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mort_diff'] > 0) &
                (subset['z_product'] >= z_threshold) &
                (subset['Mortality_Diff'] >= min_mort_diff)
            ].copy()
            
            if len(bridge_edges) > 0:
                all_bridges.append(bridge_edges)
                
    return pd.concat(all_bridges, ignore_index=True) if all_bridges else pd.DataFrame()

def add_english_descriptions_edges(df: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """Add English descriptions for edges (two codes)."""
    eng_path = base_path / 'ICD10_Diagnoses_All_ENG.csv'
    if not eng_path.exists():
        return df
        
    eng_df = pd.read_csv(eng_path)
    if 'Code' in eng_df.columns and 'ShortDescription' in eng_df.columns:
        icd_to_eng = dict(zip(eng_df['Code'], eng_df['ShortDescription']))
        
        df['Description_Eng_1'] = df['ICD_Code_1'].map(icd_to_eng).fillna(df['Description_1'])
        df['Description_Eng_2'] = df['ICD_Code_2'].map(icd_to_eng).fillna(df['Description_2'])
    
    return df

@app.command()
def bridge_edges(
    output_filename: str = "bridge_edges_mortality_ZSCORE.csv",
    top_percent: float = 5,
    min_mort_diff: float = 0.30
):
    """
    Generate the Critical Bridge Edges table (Z-score method).
    """
    logger.info("Starting Bridge Edges Analysis (Z-score Method)...")
    
    all_data = []
    
    # Load Data
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age Group {age_id} ({AGE_GROUPS[age_id]})...")
            df = load_network_edges_with_mortality(gender, age_id)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded. Check files.")
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    logger.success(f"Loaded {len(df_all)} edges for analysis.")
    
    # Identify Critical Bridges
    df_bridges = identify_critical_bridge_edges_zscore(
        df_all, 
        top_percent=top_percent, 
        min_mort_diff=min_mort_diff
    )
    
    if df_bridges.empty:
        logger.warning("No critical bridge edges identified.")
        return

    logger.success(f"Identified {len(df_bridges)} critical bridge edges.")

    # Add English Descriptions
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR
    df_bridges = add_english_descriptions_edges(df_bridges, base_data_path)
    
    # Save CSV
    out_csv = PROCESSED_DATA_DIR / output_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bridges.to_csv(out_csv, index=False)
    logger.success(f"Saved CSV to {out_csv}")
    
    # Summary
    print("\n" + "="*60)
    print("BRIDGE EDGES SUMMARY")
    print("="*60)
    print(f"Total bridge edges identified: {len(df_bridges)}")
    
    for sex in SEXES:
        sex_data = df_bridges[df_bridges['Sex'] == sex]
        print(f"\n{sex}: {len(sex_data)} edges")
        for age_id in sorted(sex_data['Age_Group'].unique()):
            age_data = sex_data[sex_data['Age_Group'] == age_id]
            if len(age_data) > 0:
                age_lbl = AGE_GROUPS[age_id]
                print(f"  {age_lbl}: {len(age_data)} edges")
                # Top 3 by mortality diff
                top3 = age_data.nlargest(3, 'Mortality_Diff')
                for _, edge in top3.iterrows():
                    # Order: lower -> higher mortality
                    if edge['Mortality_1'] < edge['Mortality_2']:
                        src, tgt = edge['ICD_Code_1'], edge['ICD_Code_2']
                        m_src, m_tgt = edge['Mortality_1'], edge['Mortality_2']
                    else:
                        src, tgt = edge['ICD_Code_2'], edge['ICD_Code_1']
                        m_src, m_tgt = edge['Mortality_2'], edge['Mortality_1']
                        
                    print(f"    - {src} -> {tgt}: Bet={edge['Edge_Betweenness']:.5f}, Mort: {m_src:.4f} -> {m_tgt:.4f} (Diff={edge['Mortality_Diff']:.4f})")

# ========================================================================================
# Visualization Functions (Added from 004_MortalityEdges_Distributions.ipynb)
# ========================================================================================

def plot_mortality_difference_histograms(df: pd.DataFrame, output_path: Path):
    """Plot histograms of mortality differences."""
    logger.info("Generating mortality difference histograms...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mortality Difference Distributions', fontsize=16, fontweight='bold')
    
    # Overall distribution
    ax = axes[0, 0]
    ax.hist(df['Mortality_Diff'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15% threshold')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20% threshold')
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('All Edges (Overall Distribution)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoomed in (0-0.3 range)
    ax = axes[0, 1]
    df_subset = df[df['Mortality_Diff'] <= 0.3]
    ax.hist(df_subset['Mortality_Diff'], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10%')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15%')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20%')
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Zoomed: 0-30% Range', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # By sex
    ax = axes[1, 0]
    for sex in ['Female', 'Male']:
        sex_data = df[df['Sex'] == sex]['Mortality_Diff']
        ax.hist(sex_data, bins=80, alpha=0.6, label=sex, edgecolor='black')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('By Sex', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 1]
    sorted_diffs = np.sort(df['Mortality_Diff'])
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    ax.plot(sorted_diffs, cumulative, linewidth=2, color='navy')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15% threshold')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20% threshold')
    
    # Add percentage labels
    for threshold in [0.10, 0.15, 0.20]:
        pct = (df['Mortality_Diff'] >= threshold).mean() * 100
        y_pos = (df['Mortality_Diff'] >= threshold).mean()
        ax.text(threshold + 0.01, y_pos, f'{pct:.1f}% above', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved histogram plot to: {output_path}")

def plot_zscore_distributions(df: pd.DataFrame, output_path: Path):
    """Plot z-score distributions for both methods."""
    logger.info("Generating z-score distributions plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Z-Score Distributions for Threshold Selection', fontsize=16, fontweight='bold')
    
    # Calculate z-scores per sex-age group
    all_zscores = []
    
    for sex in df['Sex'].unique():
        for age_group in df['Age_Group'].unique():
            subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_group)].copy()
            
            if len(subset) == 0:
                continue
            
            # Z-scores for betweenness
            bet_mean = subset['Edge_Betweenness'].mean()
            bet_std = subset['Edge_Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Edge_Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            # Z-scores for mortality diff
            mort_mean = subset['Mortality_Diff'].mean()
            mort_std = subset['Mortality_Diff'].std()
            if mort_std > 0:
                subset['z_mort_diff'] = (subset['Mortality_Diff'] - mort_mean) / mort_std
            else:
                subset['z_mort_diff'] = 0
            
            # Z-score product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            
            all_zscores.append(subset)
    
    df_z = pd.concat(all_zscores, ignore_index=True)
    
    # 1. Z-score betweenness distribution
    ax = axes[0, 0]
    ax.hist(df_z['z_betweenness'], bins=100, edgecolor='black', alpha=0.7, color='coral')
    percentiles = [90, 95, 99]
    for p in percentiles:
        val = df_z['z_betweenness'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Betweenness Z-Scores', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Z-score mortality diff distribution
    ax = axes[0, 1]
    ax.hist(df_z['z_mort_diff'], bins=100, edgecolor='black', alpha=0.7, color='lightgreen')
    for p in percentiles:
        val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Mortality Difference Z-Scores', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Z-score product distribution
    ax = axes[0, 2]
    # Only positive products
    positive_products = df_z[df_z['z_product'] > 0]['z_product']
    ax.hist(positive_products, bins=100, edgecolor='black', alpha=0.7, color='skyblue')
    for p in [90, 95, 99]:
        val = positive_products.quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score Product', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Z-Score Product (Positive Only)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution - betweenness
    ax = axes[1, 0]
    sorted_z = np.sort(df_z['z_betweenness'])
    cumulative = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    ax.plot(sorted_z, cumulative, linewidth=2, color='coral')
    for p in percentiles:
        val = df_z['z_betweenness'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}%')
        pct_above = (df_z['z_betweenness'] >= val).mean() * 100
        ax.text(val + 0.1, 0.5, f'{100-p}% above', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative: Betweenness', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution - mortality diff
    ax = axes[1, 1]
    sorted_z = np.sort(df_z['z_mort_diff'])
    cumulative = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    ax.plot(sorted_z, cumulative, linewidth=2, color='lightgreen')
    for p in percentiles:
        val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}%')
    ax.set_xlabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative: Mortality Diff', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Scatter plot: betweenness vs mortality diff z-scores
    ax = axes[1, 2]
    scatter = ax.scatter(df_z['z_betweenness'], df_z['z_mort_diff'], 
                        c=df_z['z_product'], cmap='RdYlGn', alpha=0.3, s=1)
    
    # Add percentile lines
    for p in [90, 95]:
        bet_val = df_z['z_betweenness'].quantile(p/100)
        mort_val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(bet_val, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(mort_val, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_title('Z-Score Scatter (color = product)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Z-Product')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved z-score plot to: {output_path}")

def print_threshold_statistics(df: pd.DataFrame):
    """Print statistics for different thresholds."""
    print("\n" + "="*80)
    print("MORTALITY DIFFERENCE THRESHOLD STATISTICS")
    print("="*80)
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"\n{'Threshold':>12} | {'# Edges':>10} | {'% of Total':>12} | {'Female':>8} | {'Male':>8}")
    print("-" * 80)
    
    total = len(df)
    for threshold in thresholds:
        count = (df['Mortality_Diff'] >= threshold).sum()
        pct = count / total * 100
        female_count = ((df['Mortality_Diff'] >= threshold) & (df['Sex'] == 'Female')).sum()
        male_count = ((df['Mortality_Diff'] >= threshold) & (df['Sex'] == 'Male')).sum()
        
        print(f"{threshold:11.0%} | {count:10,} | {pct:11.1f}% | {female_count:8,} | {male_count:8,}")
    
    print("\n" + "="*80)
    print("Z-SCORE PERCENTILE STATISTICS (Method 2)")
    print("="*80)
    
    # Calculate z-scores
    all_zscores = []
    for sex in df['Sex'].unique():
        for age_group in df['Age_Group'].unique():
            subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_group)].copy()
            if len(subset) == 0:
                continue
            
            bet_mean = subset['Edge_Betweenness'].mean()
            bet_std = subset['Edge_Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Edge_Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            mort_mean = subset['Mortality_Diff'].mean()
            mort_std = subset['Mortality_Diff'].std()
            if mort_std > 0:
                subset['z_mort_diff'] = (subset['Mortality_Diff'] - mort_mean) / mort_std
            else:
                subset['z_mort_diff'] = 0
            
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            all_zscores.append(subset)
    
    df_z = pd.concat(all_zscores, ignore_index=True)
    
    # Filter for positive z-products and minimum mortality diff
    df_z_filtered = df_z[(df_z['z_product'] > 0) & (df_z['Mortality_Diff'] >= 0.10)]
    
    percentiles = [80, 85, 90, 95, 99]
    
    print(f"\n{'Percentile':>12} | {'# Edges':>10} | {'% of Filtered':>15} | {'Female':>8} | {'Male':>8}")
    print("-" * 80)
    
    for p in percentiles:
        threshold_val = df_z_filtered['z_product'].quantile(p/100)
        count = (df_z_filtered['z_product'] >= threshold_val).sum()
        pct = count / len(df_z_filtered) * 100
        female_count = ((df_z_filtered['z_product'] >= threshold_val) & (df_z_filtered['Sex'] == 'Female')).sum()
        male_count = ((df_z_filtered['z_product'] >= threshold_val) & (df_z_filtered['Sex'] == 'Male')).sum()
        
        print(f"{p:11}th | {count:10,} | {pct:14.1f}% | {female_count:8,} | {male_count:8,}")

@app.command()
def visualize_thresholds():
    """
    Visualization script for mortality differences and z-scores
    Helps choose optimal thresholds for bridge edge selection.
    """
    logger.info("Starting Threshold Selection Visualization...")
    
    all_data = []
    
    # Load Data (reusing load_network_edges_with_mortality logic)
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Loading {gender} - Age Group {age_id}...")
            df = load_network_edges_with_mortality(gender, age_id)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded. Check files.")
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    logger.success(f"Loaded {len(df_all):,} edges for visualization.")
    
    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plots
    plot_mortality_difference_histograms(df_all, FIGURES_DIR / 'mortality_difference_histograms.png')
    plot_zscore_distributions(df_all, FIGURES_DIR / 'zscore_distributions.png')
    
    # Statistics
    print_threshold_statistics(df_all)
    
    logger.success("Visualization Complete.")

# ========================================================================================
# Overlap Analysis Functions (Added from 006_OverlapFinalTable.ipynb)
# ========================================================================================

def find_intersection(df_outliers: pd.DataFrame, df_high_mort_bet: pd.DataFrame) -> pd.DataFrame:
    """Find intersection of degree outliers and high mortality/betweenness nodes"""
    print("\n" + "="*80)
    print("FINDING INTERSECTION")
    print("="*80)
    
    # Normalize Age_Group in outliers to match high_mort_bet (which uses ints)
    # Check if Age_Group is string and contains 'age_'
    if df_outliers['Age_Group'].dtype == object and df_outliers['Age_Group'].str.contains('age_').any():
         df_outliers = df_outliers.copy() # Avoid SettingWithCopyWarning
         df_outliers['Age_Group_Int'] = df_outliers['Age_Group'].str.replace('age_', '').astype(int)
    else:
         df_outliers['Age_Group_Int'] = df_outliers['Age_Group']

    # Create unique identifiers
    # Format: Sex_Age_ICD
    df_outliers['node_id'] = df_outliers['Sex'] + '_' + df_outliers['Age_Group_Int'].astype(str) + '_' + df_outliers['ICD_Code']
    
    # high_mort_bet already has int Age_Group from load_network_and_mortality
    df_high_mort_bet['node_id'] = df_high_mort_bet['Sex'] + '_' + df_high_mort_bet['Age_Group'].astype(str) + '_' + df_high_mort_bet['ICD_Code']
    
    # Find intersection
    intersection_ids = set(df_outliers['node_id']) & set(df_high_mort_bet['node_id'])
    
    print(f"\nDegree outliers (high): {len(df_outliers)}")
    print(f"High mortality + betweenness (Z-score): {len(df_high_mort_bet)}")
    print(f"Intersection: {len(intersection_ids)}")
    
    if not intersection_ids:
        return pd.DataFrame()

    # Get full data for intersection (prioritizing high_mort_bet columns)
    df_intersection = df_high_mort_bet[df_high_mort_bet['node_id'].isin(intersection_ids)].copy()
    
    # Merge with outlier data to get Log_ratio and Prevalence
    # Only merging columns present in the notebook code to ensure identical output structure
    outlier_cols = ['node_id', 'Log_ratio', 'Prevalence']
    
    outlier_subset = df_outliers[outlier_cols].rename(columns={'Log_ratio': 'Log_Ratio'})
    
    df_intersection = df_intersection.merge(outlier_subset, on='node_id', how='left')
    
    # Sort
    df_intersection = df_intersection.sort_values(['Sex', 'Age_Group', 'z_geom_mean'], ascending=[True, True, False])
    
    return df_intersection

@app.command()
def critical_nodes_pipeline(
    top_percent_sinks: int = 40
):
    """
    Complete pipeline to generate Critical Nodes Intersection (Z-Score method).
    Mirrors logic from 006_OverlapFinalTable.ipynb (Cells 20 & 21).
    Generates:
      1. Degree_Prevalence_ICD_raw_EXACT.csv
      2. Degree_Prevalence_ICD_EXACT.csv (Outlier detection)
      3. Outliers_EXACT.csv
      4. critical_nodes_intersection_ZSCORE.csv
    """
    logger.info("Starting Critical Nodes Pipeline...")
    
    # ==============================================================================
    # STEP 1: LOAD DEGREE AND PREVALENCE DATA (CELL 20 Logic)
    # ==============================================================================
    print("="*80)
    print("OUTLIER DETECTION - EXACT NOTEBOOK REPLICATION")
    print("="*80)
    print("\nStep 1: Building degree-prevalence dataframe...")
    
    all_data_deg_prev = []
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            print(f"Loading {gender} age_{age_id}")
            df = load_network_and_prevalence(gender, age_id)
            if not df.empty:
                all_data_deg_prev.append(df)
    
    if not all_data_deg_prev:
        logger.error("No data available.")
        raise typer.Exit(code=1)
        
    df_deg_prev = pd.concat(all_data_deg_prev, ignore_index=True)
    print(f"Total records: {len(df_deg_prev)}")
    
    # Save raw data
    raw_file = PROCESSED_DATA_DIR / 'Degree_Prevalence_ICD_raw_EXACT.csv'
    df_deg_prev.to_csv(raw_file, index=False)
    print(f"Raw data saved to: {raw_file}")
    
    # ==============================================================================
    # STEP 2: FIND OUTLIERS (Exact Method from Cell 20)
    # ==============================================================================
    print("\nStep 2: Finding outliers using 5th/95th percentile method...")
    
    # Exact outlier detection using modified z-score
    df_outliers_processed = detect_outliers_exact(df_deg_prev)
    
    # Save processed data
    processed_file = PROCESSED_DATA_DIR / 'Degree_Prevalence_ICD_EXACT.csv'
    df_outliers_processed.to_csv(processed_file, index=False)
    print(f"Processed data saved to: {processed_file}")
    
    # Step 3: Filter to only outliers
    # The notebook filters for Outlier==True and positive Deviation for high degree
    df_outliers_only = df_outliers_processed[df_outliers_processed['Outlier'] == True].copy()
    
    outliers_file = PROCESSED_DATA_DIR / 'Outliers_EXACT.csv'
    df_outliers_only.to_csv(outliers_file, index=False)
    print(f"Outliers saved to: {outliers_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal diseases analyzed: {len(df_outliers_processed)}")
    print(f"Total outliers identified: {len(df_outliers_only)}")
    
    # ==============================================================================
    # STEP 3: INTERSECTION ANALYSIS (CELL 21 Logic)
    # ==============================================================================
    print("\n" + "="*80)
    print("INTERSECTION ANALYSIS: DEGREE OUTLIERS × HIGH MORTALITY/BETWEENNESS")
    print("="*80)
    
    # Filter for high degree outliers only (positive deviation) as done in notebook cell 21
    if 'Deviation' in df_outliers_only.columns:
        df_high_degree_outliers = df_outliers_only[df_outliers_only['Deviation'] > 0].copy()
    else:
        # Fallback if deviation not present (should be there from exact detection)
        df_high_degree_outliers = df_outliers_only.copy()
        
    print(f"Loaded {len(df_high_degree_outliers)} high degree outliers (95th percentile)")

    # Calculate High Mortality Sinks (Z-score)
    print(f"\n2. Calculating high mortality & betweenness nodes (Z-score method, top {top_percent_sinks}%)...")
    
    all_data_sinks = []
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            df = load_network_and_mortality(gender, age_id)
            if not df.empty:
                all_data_sinks.append(df)
                
    if not all_data_sinks:
        print("No data for sinks analysis.")
        return
        
    df_all_sinks = pd.concat(all_data_sinks, ignore_index=True)
    df_sinks = identify_high_mortality_sinks_zscore(df_all_sinks, top_percent=top_percent_sinks)
    print(f"Found {len(df_sinks)} nodes with high mortality & betweenness")

    # Find Intersection
    print("\n3. Finding intersection...")
    df_intersection = find_intersection(df_high_degree_outliers, df_sinks)
    
    if df_intersection.empty:
        print("\nNo intersection found!")
        return

    # Add Descriptions
    print("\n4. Adding English descriptions...")
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR
    df_intersection = add_english_descriptions(df_intersection, base_data_path)

    # Save
    print("\n6. Saving outputs...")
    out_path = PROCESSED_DATA_DIR / 'critical_nodes_intersection_ZSCORE.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_intersection.to_csv(out_path, index=False)
    print(f"✓ Data CSV saved to: {out_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: CRITICAL NODES (INTERSECTION)")
    print("="*60)
    print(f"Total critical nodes: {len(df_intersection)}")
    
    for sex in SEXES:
        sex_data = df_intersection[df_intersection['Sex'] == sex]
        print(f"\n{sex}: {len(sex_data)} nodes")
        for age_id in sorted(sex_data['Age_Group'].unique()):
            age_data = sex_data[sex_data['Age_Group'] == age_id]
            if len(age_data) > 0:
                age_lbl = AGE_GROUPS[age_id]
                print(f"  {age_lbl}: {len(age_data)} nodes")
                for _, row in age_data.iterrows():
                    code = row['ICD_Code']
                    z_geo = row.get('z_geom_mean', 0)
                    bet = row.get('Betweenness', 0)
                    mort = row.get('Mortality', 0)
                    log_r = row.get('Log_Ratio', 0)
                    print(f"    - {code} (Z-GeoMean={z_geo:.3f}, Bet={bet:.5f}, Mort={mort:.4f}, LogRatio={log_r:.2f})")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)

@app.command()
def main(
    output_filename: str = "outliers_data_S1.csv"
):
    """
    Main entry point for outlier detection.
    Loads data, calculates degrees/prevalence, detects outliers, and saves CSV.
    """
    logger.info("Starting Standalone Outlier Detection")
    
    all_data = []
    
    # Loop over Sexes and Age Groups defined in config.py
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age Group {age_id} ({AGE_GROUPS[age_id]})...")
            df = load_network_and_prevalence(gender, age_id)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded. Check file paths.")
        raise typer.Exit(code=1)

    df_all = pd.concat(all_data, ignore_index=True)
    logger.success(f"Loaded {len(df_all)} total disease-age-sex combinations with Degree>0 & Prev>0.")
    
    # Detect Outliers
    df_outliers = detect_outliers_logic(df_all)
    logger.success(f"Detected {len(df_outliers)} initial outliers.")
    
    # Add Descriptions
    # Define base path again for description loading
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR
        
    df_outliers = add_english_descriptions(df_outliers, base_data_path)

    # Select Top Outliers (Filter)
    df_outliers = select_top_outliers(df_outliers)
    logger.success(f"Selected top outliers for final table: {len(df_outliers)} rows.")
    
    # Save Results
    output_path = PROCESSED_DATA_DIR / output_filename
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_outliers.to_csv(output_path, index=False)
    logger.success(f"Outlier analysis saved to: {output_path}")

    # Print Summary (reproducing notebook summary)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for sex in SEXES:
        print(f"\n{sex}:")
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        for age_label in sorted(sex_data['Age_Range'].unique()):
            count = len(sex_data[sex_data['Age_Range'] == age_label])
            print(f"  {age_label:5}: {count} outliers")
        print(f"  TOTAL: {len(sex_data)}")

if __name__ == "__main__":
    app()