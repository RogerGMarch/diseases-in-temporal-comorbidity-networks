"""
Bridge Edge Analysis: High Betweenness + Large Mortality Difference.

This script implements the methodology from '003_MortalityEdges.ipynb' to:
1. Identify critical bridge edges using Percentile thresholds (Method 1).
2. Identify critical bridge edges using Z-score product method (Method 2).

It uses the enhanced `NetworkAnalyzer` in `features.py` to load edge metrics,
calculates statistics, filters for critical bridges, and exports results to CSV.
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()


def identify_critical_bridge_edges(
    df_all: pd.DataFrame, 
    bet_percentile: int = 95, 
    mort_diff_percentile: int = 95, 
    min_mort_diff: float = 0.10
) -> pd.DataFrame:
    """
    Method 1: Identify edges with high betweenness AND large mortality difference.
    
    Criteria:
    1. Edge Betweenness > Xth percentile (default 95)
    2. Mortality Difference > Yth percentile (default 95)
    3. Minimum Absolute Mortality Difference > min_mort_diff (default 0.10)
    """
    logger.info("METHOD 1: PERCENTILE THRESHOLDS")
    logger.info(f"  - Betweenness > {bet_percentile}th percentile")
    logger.info(f"  - Mortality difference > {mort_diff_percentile}th percentile")
    logger.info(f"  - Minimum absolute mortality difference > {min_mort_diff*100:.0f}%")
    
    all_bridges = []
    
    for sex in SEXES:
        for age_group_id in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate thresholds
            bet_threshold = subset['Edge_Betweenness'].quantile(bet_percentile / 100)
            mort_diff_threshold = subset['Mortality_Diff'].quantile(mort_diff_percentile / 100)
            
            # Identify bridge edges
            bridge_edges = subset[
                (subset['Edge_Betweenness'] >= bet_threshold) & 
                (subset['Mortality_Diff'] >= mort_diff_threshold) &
                (subset['Mortality_Diff'] >= min_mort_diff)
            ].copy()
            
            if len(bridge_edges) > 0:
                bridge_edges['Betweenness_Percentile'] = bet_percentile
                bridge_edges['Mort_Diff_Percentile'] = mort_diff_percentile
                all_bridges.append(bridge_edges)
    
    if not all_bridges:
        return pd.DataFrame()
    
    return pd.concat(all_bridges, ignore_index=True)


def identify_critical_bridge_edges_zscore(
    df_all: pd.DataFrame, 
    top_percent: int = 40, 
    min_mort_diff: float = 0.10
) -> pd.DataFrame:
    """
    Method 2: Identify bridge edges using Z-score product method (Manuscript approach).
    
    Steps:
    1. Compute z(betweenness) and z(mortality_diff).
    2. Calculate product: z(betweenness) * z(mortality_diff).
    3. Select top X% of positive products.
    4. Enforce minimum mortality difference.
    """
    logger.info("METHOD 2: Z-SCORE PRODUCT")
    logger.info(f"  - Computing z(betweenness) × z(mortality_diff)")
    logger.info(f"  - Selecting top {top_percent}% of z-score products")
    logger.info(f"  - Minimum absolute mortality difference > {min_mort_diff*100:.0f}%")
    
    all_bridges = []
    
    for sex in SEXES:
        for age_group_id in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate z-scores for betweenness
            bet_mean = subset['Edge_Betweenness'].mean()
            bet_std = subset['Edge_Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Edge_Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            # Calculate z-scores for mortality difference
            mort_mean = subset['Mortality_Diff'].mean()
            mort_std = subset['Mortality_Diff'].std()
            if mort_std > 0:
                subset['z_mort_diff'] = (subset['Mortality_Diff'] - mort_mean) / mort_std
            else:
                subset['z_mort_diff'] = 0
            
            # Calculate z-score product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            
            # Filter: positive z-scores, above threshold percentile, and minimum mortality diff
            threshold_percentile = 100 - top_percent
            z_threshold = subset['z_product'].quantile(threshold_percentile / 100)
            
            bridge_edges = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mort_diff'] > 0) &
                (subset['z_product'] >= z_threshold) &
                (subset['Mortality_Diff'] >= min_mort_diff)
            ].copy()
            
            if len(bridge_edges) > 0:
                bridge_edges['Selection_Method'] = 'Z-Score Product'
                all_bridges.append(bridge_edges)
    
    if not all_bridges:
        return pd.DataFrame()
    
    return pd.concat(all_bridges, ignore_index=True)


@app.command()
def main():
    """Execute Critical Bridge Edge Analysis."""
    logger.info("Starting Critical Bridge Edge Analysis Pipeline")
    
    # Step 1: Load all data
    logger.info("Step 1: Loading edge data with mortality...")
    all_data = []
    
    for sex in SEXES:
        for age_id, age_range in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex} data"):
            # Use the updated load_edge_metrics from features.py
            # This handles all file loading, betweenness calculation, and mortality mapping
            df = NetworkAnalyzer.load_edge_metrics(sex, age_id)
            
            if not df.empty:
                all_data.append(df)
            else:
                logger.warning(f"No edge data found for {sex} Age {age_id}")
    
    if not all_data:
        logger.error("No data loaded. Check data paths.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total edges analyzed: {len(df_all)}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Method 1 (Percentile)
    # Note: Using min_mort_diff=0.30 as per notebook main execution
    df_bridges_percentile = identify_critical_bridge_edges(
        df_all, 
        bet_percentile=95, 
        mort_diff_percentile=95,
        min_mort_diff=0.30
    )
    
    if not df_bridges_percentile.empty:
        # Add English descriptions
        df_bridges_percentile = NetworkAnalyzer.add_english_descriptions(df_bridges_percentile)
        
        # Save CSV
        csv_file_1 = PROCESSED_DATA_DIR / 'bridge_edges_mortality_PERCENTILE.csv'
        df_bridges_percentile.to_csv(csv_file_1, index=False)
        logger.success(f"✓ Method 1 (Percentile) CSV saved to: {csv_file_1}")
        logger.info(f"  - Critical bridge edges identified: {len(df_bridges_percentile)}")
    else:
        logger.warning("No edges found for Method 1 (Percentile).")

    # Step 3: Method 2 (Z-Score)
    # Note: Using top_percent=5 and min_mort_diff=0.30 as per notebook main execution
    df_bridges_zscore = identify_critical_bridge_edges_zscore(
        df_all,
        top_percent=5, 
        min_mort_diff=0.30
    )
    
    if not df_bridges_zscore.empty:
        # Add English descriptions
        df_bridges_zscore = NetworkAnalyzer.add_english_descriptions(df_bridges_zscore)
        
        # Save CSV
        csv_file_2 = PROCESSED_DATA_DIR / 'bridge_edges_mortality_ZSCORE.csv'
        df_bridges_zscore.to_csv(csv_file_2, index=False)
        logger.success(f"✓ Method 2 (Z-Score) CSV saved to: {csv_file_2}")
        logger.info(f"  - Critical bridge edges identified: {len(df_bridges_zscore)}")
    else:
        logger.warning("No edges found for Method 2 (Z-Score).")

    # Comparison and Summary
    logger.info("\n" + "="*40)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*40)
    
    count_1 = len(df_bridges_percentile) if not df_bridges_percentile.empty else 0
    count_2 = len(df_bridges_zscore) if not df_bridges_zscore.empty else 0
    
    logger.info(f"Method 1 (Percentile 95th): {count_1} edges")
    logger.info(f"Method 2 (Z-Score top 5%):  {count_2} edges")

    if count_1 > 0 and count_2 > 0:
        # Create edge identifiers for comparison
        df_bridges_percentile['edge_id'] = df_bridges_percentile.apply(
            lambda x: f"{x['Sex']}_{x['Age_Group']}_{x['ICD_Code_1']}_{x['ICD_Code_2']}", axis=1
        )
        df_bridges_zscore['edge_id'] = df_bridges_zscore.apply(
            lambda x: f"{x['Sex']}_{x['Age_Group']}_{x['ICD_Code_1']}_{x['ICD_Code_2']}", axis=1
        )
        
        overlap = set(df_bridges_percentile['edge_id']) & set(df_bridges_zscore['edge_id'])
        logger.info(f"Overlapping edges: {len(overlap)}")
        logger.info(f"Overlap percentage: {len(overlap)/count_1*100:.1f}% of Method 1")


if __name__ == '__main__':
    app()