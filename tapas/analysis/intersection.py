"""
Intersection Analysis: Degree Outliers × High Mortality/Betweenness.

This script replicates the logic of '006_OverlapFinalTable.ipynb':
1. Identify "Degree Outliers" (using 20th/80th percentiles via detect_outliers_exact)
2. Identify "High Mortality & Betweenness" nodes using the Z-score product method
3. Find the intersection of these two sets
4. Generate the final critical nodes CSV
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

# IMPORT THE NEW EXACT FUNCTION
from tapas.analysis.outliers import detect_outliers_exact

app = typer.Typer()

def calculate_high_mortality_betweenness_nodes(df_all: pd.DataFrame, top_percent: int = 40) -> pd.DataFrame:
    """
    Identify nodes with high mortality AND high betweenness using Z-score product method.
    """
    logger.info(f"Calculating high mortality & betweenness nodes (Top {top_percent}% Z-score product)...")
    
    high_nodes = []
    
    for sex in SEXES:
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group)
            ].copy()
            
            if len(subset) == 0:
                continue
                
            # Calculate Z-scores
            bet_std = subset['Betweenness'].std()
            mort_std = subset['Mortality'].std()
            
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Betweenness'] - subset['Betweenness'].mean()) / bet_std
            else:
                subset['z_betweenness'] = 0
                
            if mort_std > 0:
                subset['z_mortality'] = (subset['Mortality'] - subset['Mortality'].mean()) / mort_std
            else:
                subset['z_mortality'] = 0
            
            # Calculate Product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mortality']
            
            # Calculate Geometric Mean for ranking
            subset['z_geom_mean'] = np.where(
                (subset['z_betweenness'] > 0) & (subset['z_mortality'] > 0),
                np.sqrt(subset['z_betweenness'] * subset['z_mortality']),
                0
            )
            
            # Filter: Positive Z-scores and top X%
            threshold_percentile = 100 - top_percent
            z_threshold = subset['z_product'].quantile(threshold_percentile / 100)
            
            high_subset = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mortality'] > 0) &
                (subset['z_product'] >= z_threshold)
            ].copy()
            
            if len(high_subset) > 0:
                high_nodes.append(high_subset)
    
    if not high_nodes:
        return pd.DataFrame()
        
    return pd.concat(high_nodes, ignore_index=True)


def find_intersection(df_outliers: pd.DataFrame, df_zscore: pd.DataFrame) -> pd.DataFrame:
    """
    Find intersection between Degree Outliers and High Mortality/Betweenness nodes.
    Matches on Sex, Age_Group, and ICD_Code.
    """
    logger.info("Finding intersection...")
    
    # Create unique keys for joining
    def create_key(df):
        return df['Sex'] + '_' + df['Age_Group'].astype(str) + '_' + df['ICD_Code']

    df_outliers = df_outliers.copy()
    df_zscore = df_zscore.copy()
    
    df_outliers['node_key'] = create_key(df_outliers)
    df_zscore['node_key'] = create_key(df_zscore)
    
    # Intersection keys
    keys_outliers = set(df_outliers['node_key'])
    keys_zscore = set(df_zscore['node_key'])
    intersection_keys = keys_outliers.intersection(keys_zscore)
    
    logger.info(f"Degree Outliers: {len(keys_outliers)}")
    logger.info(f"High Z-Score Nodes: {len(keys_zscore)}")
    logger.info(f"Intersection: {len(intersection_keys)}")
    
    # Filter Z-score dataframe to intersection
    df_intersection = df_zscore[df_zscore['node_key'].isin(intersection_keys)].copy()
    
    # Merge relevant fields from outliers
    cols_to_merge = ['node_key', 'Log_ratio', 'Ratio', 'Deviation']
    available_cols = [c for c in cols_to_merge if c in df_outliers.columns]
    
    df_merge = df_outliers[available_cols].drop_duplicates()
    
    df_final = df_intersection.merge(df_merge, on='node_key', how='left')
    
    # Sort by Sex, Age, Z-Geom-Mean (descending)
    if not df_final.empty:
        df_final = df_final.sort_values(
            ['Sex', 'Age_Group', 'z_geom_mean'], 
            ascending=[True, True, False]
        )
    
    return df_final


@app.command()
def main():
    """Main execution flow for Intersection Analysis."""
    logger.info("="*60)
    logger.info("INTERSECTION ANALYSIS: DEGREE OUTLIERS × HIGH MORTALITY/BETWEENNESS")
    logger.info("="*60)

    # 1. Load All Data (Metric calculation)
    logger.info("1. Loading base metrics for all nodes...")
    all_data_list = []
    for sex in SEXES:
        for age_id, _ in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex}"):
            df = NetworkAnalyzer.load_node_metrics(sex, age_id)
            if not df.empty:
                all_data_list.append(df)
    
    if not all_data_list:
        logger.error("No data loaded.")
        return
    
    df_all = pd.concat(all_data_list, ignore_index=True)
    df_all = NetworkAnalyzer.add_english_descriptions(df_all)
    
    # 2. Get Degree Outliers using EXACT METHOD (20th/80th percentiles)
    logger.info("2. Identifying Degree Outliers (Using EXACT notebook method: 20/80)...")
    
    # CALLING THE EXACT FUNCTION
    df_processed = detect_outliers_exact(df_all)
    
    if 'Log_ratio' not in df_processed.columns:
        logger.error("Error: detect_outliers_exact failed to create 'Log_ratio' column.")
        return

    # Filter for HIGH degree outliers (Deviation > 0 and Outlier == True)
    df_degree_outliers = df_processed[
        (df_processed['Outlier'] == True) & 
        (df_processed['Deviation'] > 0)
    ].copy()
    
    logger.info(f"Found {len(df_degree_outliers)} High Degree Outliers.")

    # 3. Get High Mortality/Betweenness Nodes
    logger.info("3. Identifying High Mortality & Betweenness Nodes (Top 40%)...")
    df_zscore = calculate_high_mortality_betweenness_nodes(df_processed, top_percent=40)
    logger.info(f"Found {len(df_zscore)} High Z-Score Nodes.")

    # 4. Find Intersection
    df_intersection = find_intersection(df_degree_outliers, df_zscore)

    if df_intersection.empty:
        logger.warning("No intersection found.")
        return

    # 5. Save Output
    output_file = PROCESSED_DATA_DIR / 'critical_nodes_intersection_ZSCORE.csv'
    
    final_cols = [
        'Sex', 'Age_Group', 'Age_Range', 'ICD_Code', 
        'Description_GER', 'Description_Eng', 
        'Degree', 'Betweenness', 'Mortality', 
        'z_product', 'z_geom_mean', 'Log_ratio'
    ]
    final_cols = [c for c in final_cols if c in df_intersection.columns]
    
    df_intersection[final_cols].to_csv(output_file, index=False)
    logger.success(f"✓ Analysis Complete. Saved to: {output_file}")
    
    # Print Summary
    logger.info("\nSUMMARY: Critical Nodes (Intersection)")
    logger.info(f"Total: {len(df_intersection)}")
    
    for sex in SEXES:
        sex_df = df_intersection[df_intersection['Sex'] == sex]
        logger.info(f"{sex}: {len(sex_df)} nodes")
        if len(sex_df) > 0:
            for _, row in sex_df.head(3).iterrows():
                icd = row.get('ICD_Code', 'N/A')
                age = row.get('Age_Range', 'N/A')
                z_geom = row.get('z_geom_mean', 0.0)
                log_r = row.get('Log_ratio', 0.0)
                logger.info(f"  - {icd} ({age}): Z-GeoMean={z_geom:.3f}, LogRatio={log_r:.2f}")

if __name__ == "__main__":
    app()