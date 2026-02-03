import pandas as pd
import numpy as np
from pathlib import Path
import typer
from loguru import logger
from typing import Optional

# Import project configuration and features
from tapas.config import (
    DATA_DIR, 
    PROCESSED_DATA_DIR, 
    INTERIM_DATA_DIR,
    AGE_GROUPS, 
    SEXES,
    REPORTS_DIR
)
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def format_prevalence_scientific(val):
    """Format prevalence in scientific notation (helper for potential latex output later)."""
    if val == 0 or pd.isna(val) or val < 1e-5:
        return "$< 10^{-5}$"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10 ** exponent)
    if abs(mantissa - 1.0) < 0.01:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"

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

def detect_outliers_logic(df_all: pd.DataFrame) -> pd.DataFrame:
    """Apply 20th/80th percentile thresholding on Log Ratio."""
    logger.info("Detecting outliers using 20th/80th percentile method...")
    
    df_all['Ratio'] = df_all['Degree'] / df_all['Prevalence']
    df_all['Log_ratio'] = df_all['Ratio'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    all_outliers = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            lower_bound = subset['Log_ratio'].quantile(0.5)
            upper_bound = subset['Log_ratio'].quantile(0.95)
            
            subset['Outlier'] = (
                (subset['Log_ratio'] < lower_bound) | 
                (subset['Log_ratio'] > upper_bound)
            )
            
            outliers = subset[subset['Outlier'] == True]
            if len(outliers) > 0:
                all_outliers.append(outliers)
                
    if not all_outliers:
        return pd.DataFrame()
        
    result_df = pd.concat(all_outliers, ignore_index=True)
    return result_df.sort_values(
        by=['Sex', 'Age_Group', 'Degree'], 
        ascending=[True, True, False]
    )

def add_english_descriptions(df_outliers: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    """Add English descriptions mapping."""
    logger.info("Adding English descriptions...")
    
    eng_path = base_path / 'ICD10_Diagnoses_All_ENG.csv'
    
    if not eng_path.exists():
        logger.warning(f"English description file not found at {eng_path}. Skipping translation.")
        df_outliers['Description_Eng'] = df_outliers['Description_GER']
        return df_outliers

    eng_df = pd.read_csv(eng_path)
    if 'Code' in eng_df.columns and 'ShortDescription' in eng_df.columns:
        icd_to_eng = dict(zip(eng_df['Code'], eng_df['ShortDescription']))
        df_outliers['Description_Eng'] = df_outliers['ICD_Code'].map(icd_to_eng)
        df_outliers['Description_Eng'] = df_outliers['Description_Eng'].fillna(df_outliers['Description_GER'])
    else:
        logger.warning("English dictionary columns format not recognized.")
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
# Mortality Sinks Functions (Added from 002_HighMortalityHighBetweenesNodes.ipynb)
# ========================================================================================

def load_network_and_mortality(gender: str, age_group_id: int) -> pd.DataFrame:
    """
    Load network data and mortality data to calculate betweenness and mortality.
    Mirroring the logic from load_network_with_mortality in the notebook.
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
        # Calculate betweenness using normalized=True (default in nx and notebook)
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
    Calculates z(bet) * z(mort) and selects top 20% of positive products.
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
    Generate the High Mortality Sinks table (Z-score method) 
    mirroring the methodology in 002_HighMortalityHighBetweenesNodes.ipynb.
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