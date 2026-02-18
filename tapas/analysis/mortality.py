"""
Mortality and Centrality Analysis.

This script implements the methodology from '002_HighMortalityHighBetweenesNodes.ipynb' to:
1. Identify critical nodes with high Betweenness Centrality AND High Mortality (80th percentile rule).
2. Identify "High-Mortality Sinks" using the Z-score product method (Manuscript methodology).

It reuses the robust data loading from `features.py` and exports the results to CSV.
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()


def identify_high_betweenness_high_mortality(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Identify nodes with high betweenness AND high mortality.
    
    Rule: Node must be >= 80th percentile for Betweenness AND 
          >= 80th percentile for Mortality within its Sex/Age group.
    """
    logger.info("Identifying nodes with high betweenness and high mortality (80th percentile)...")
    
    all_high = []
    
    # Process each sex-age group separately
    for sex in SEXES:
        for age_group_id in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate thresholds (80th percentile for both as per notebook logic)
            bet_threshold = subset['Betweenness'].quantile(0.80)
            mort_threshold = subset['Mortality'].quantile(0.80)
            
            # Identify nodes with BOTH high betweenness AND high mortality
            high_nodes = subset[
                (subset['Betweenness'] >= bet_threshold) & 
                (subset['Mortality'] >= mort_threshold)
            ].copy()
            
            if len(high_nodes) > 0:
                # Notebook hardcodes these to 90 despite using 0.80 quantile
                high_nodes['Betweenness_Percentile'] = 90
                high_nodes['Mortality_Percentile'] = 90
                all_high.append(high_nodes)
    
    if not all_high:
        return pd.DataFrame()
        
    return pd.concat(all_high, ignore_index=True)


def identify_high_mortality_sinks_zscore(df_all: pd.DataFrame, top_percent: int = 20) -> pd.DataFrame:
    """
    Identify high-mortality sinks using Z-score product method (Manuscript methodology).
    
    Method:
    1. Calculate Z-scores for Betweenness and Mortality per group.
    2. Product = Z(Betweenness) * Z(Mortality).
    3. Select top X% of positive products.
    """
    logger.info(f"Identifying high-mortality sinks (Z-Score method, top {top_percent}%)...")
    
    all_high = []
    
    # Process each sex-age group separately
    for sex in SEXES:
        for age_group_id in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & 
                (df_all['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate z-scores for betweenness
            bet_mean = subset['Betweenness'].mean()
            bet_std = subset['Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            # Calculate z-scores for mortality
            mort_mean = subset['Mortality'].mean()
            mort_std = subset['Mortality'].std()
            if mort_std > 0:
                subset['z_mortality'] = (subset['Mortality'] - mort_mean) / mort_std
            else:
                subset['z_mortality'] = 0
            
            # Calculate z-score product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mortality']
            
            # Calculate geometric mean (for reporting) - Only for positive z-scores
            # NOTE: np.where evaluates both branches. If product is negative (e.g. pos * neg),
            # np.sqrt will raise a RuntimeWarning even if we filter it out in the condition.
            # Fix: Clip product to 0 before sqrt.
            subset['z_geom_mean'] = np.where(
                (subset['z_betweenness'] > 0) & (subset['z_mortality'] > 0),
                np.sqrt((subset['z_betweenness'] * subset['z_mortality']).clip(lower=0)),
                0
            )
            
            # Filter: positive z-scores and top X% of product
            threshold_percentile = 100 - top_percent
            z_threshold = subset['z_product'].quantile(threshold_percentile / 100)
            
            high_nodes = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mortality'] > 0) &
                (subset['z_product'] >= z_threshold)
            ].copy()
            
            if len(high_nodes) > 0:
                all_high.append(high_nodes)
    
    if not all_high:
        return pd.DataFrame()
        
    return pd.concat(all_high, ignore_index=True)


@app.command()
def main():
    """Execute Mortality and Betweenness Analysis."""
    logger.info("Starting High Betweenness + High Mortality Analysis Pipeline")
    
    # Step 1: Load all data
    logger.info("Step 1: Loading network and mortality data...")
    all_data = []
    
    for sex in SEXES:
        for age_id, age_range in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex} data"):
            # features.py already loads Betweenness, Mortality, and ICD descriptions
            df = NetworkAnalyzer.load_node_metrics(sex, age_id)
            
            if not df.empty:
                # Notebook filters: degree > 0
                df = df[df['Degree'] > 0].copy()
                all_data.append(df)
            else:
                logger.warning(f"No data found for {sex} Age {age_id}")
    
    if not all_data:
        logger.error("No data loaded. Check data paths.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total connected nodes analyzed: {len(df_all)}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Analysis A - 80th Percentile Cutoff
    df_high_percentile = identify_high_betweenness_high_mortality(df_all)
    
    if not df_high_percentile.empty:
        # Add descriptions if missing (NetworkAnalyzer usually adds GER, we want Eng)
        df_high_percentile = NetworkAnalyzer.add_english_descriptions(df_high_percentile)
        
        # Sort for consistency
        df_high_percentile = df_high_percentile.sort_values(
            ['Sex', 'Age_Group', 'Betweenness'], ascending=[True, True, False]
        )

        csv_file_a = PROCESSED_DATA_DIR / 'high_betweenness_mortality.csv'
        
        # Select relevant columns
        cols_a = [
            'Sex', 'Age_Group', 'Age_Range', 'ICD_Code', 
            'Description_GER', 'Description_Eng', 
            'Degree', 'Betweenness', 'Mortality', 
            'Betweenness_Percentile', 'Mortality_Percentile'
        ]
        cols_a = [c for c in cols_a if c in df_high_percentile.columns]
        
        df_high_percentile[cols_a].to_csv(csv_file_a, index=False)
        logger.success(f"✓ High Betweenness/Mortality (80th pct) saved to: {csv_file_a}")
        logger.info(f"  - identified {len(df_high_percentile)} nodes")
    else:
        logger.warning("No nodes found for 80th percentile analysis.")

    # Step 3: Analysis B - Z-Score Sinks
    df_sinks = identify_high_mortality_sinks_zscore(df_all, top_percent=20)
    
    if not df_sinks.empty:
        df_sinks = NetworkAnalyzer.add_english_descriptions(df_sinks)
        
        # Sort by Geometric Mean
        df_sinks = df_sinks.sort_values(
            ['Sex', 'Age_Group', 'z_geom_mean'], ascending=[True, True, False]
        )
        
        csv_file_b = PROCESSED_DATA_DIR / 'high_mortality_sinks_ZSCORE.csv'
        
        cols_b = [
            'Sex', 'Age_Group', 'Age_Range', 'ICD_Code', 
            'Description_GER', 'Description_Eng', 
            'Degree', 'Betweenness', 'Mortality', 
            'z_betweenness', 'z_mortality', 'z_product', 'z_geom_mean'
        ]
        cols_b = [c for c in cols_b if c in df_sinks.columns]
        
        df_sinks[cols_b].to_csv(csv_file_b, index=False)
        logger.success(f"✓ High Mortality Sinks (Z-Score) saved to: {csv_file_b}")
        logger.info(f"  - identified {len(df_sinks)} sinks")
    else:
        logger.warning("No sinks found for Z-score analysis.")

    logger.info("\n" + "="*40)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*40)
    
    if not df_high_percentile.empty:
        logger.info(f"High Bet/Mort (80th pct): {len(df_high_percentile)} nodes")
    if not df_sinks.empty:
        logger.info(f"Z-Score Sinks: {len(df_sinks)} nodes")

if __name__ == '__main__':
    app()