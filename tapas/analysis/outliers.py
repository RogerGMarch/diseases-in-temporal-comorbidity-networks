"""
Outlier detection for comorbidity networks.

This module consolidates outlier analysis functionality including:
1. Degree-prevalence outlier detection (high/low degree relative to prevalence)
2. High-mortality sinks identification (high betweenness + high mortality)

Paper References:
- Degree-prevalence outliers: Supplementary Table S1
- High-mortality sinks: Main text, Section on critical disease nodes
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer
from tapas.utils.statistics import modified_zscore, compute_z_score

app = typer.Typer()


def detect_outliers_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using 20th/80th percentile thresholds.
    
    Identifies diseases with unusually high or low degree relative to their
    prevalence using log-ratio analysis.
    
    Methodology:
    1. Calculate ratio = degree / prevalence
    2. Calculate log10(ratio) to handle wide range of values
    3. Identify outliers using 20th/80th percentile thresholds
    4. Calculate modified z-score using MAD for robustness
    
    Args:
        df: DataFrame with columns: Sex, Age_Group, ICD_Code, Degree, Prevalence
        
    Returns:
        DataFrame with added columns:
        - Ratio: degree / prevalence
        - Log_ratio: log10(ratio)
        - Deviation: modified z-score
        - Outlier: True if below 20th or above 80th percentile
        
    Examples:
        >>> df = load_network_data()
        >>> outliers_df = detect_outliers_exact(df)
        >>> high_degree_outliers = outliers_df[outliers_df['Outlier'] & (outliers_df['Log_ratio'] > median)]
    """
    logger.info("Detecting outliers (20th/80th percentile)...")
    
    df['Ratio'] = df['Degree'] / df['Prevalence']
    df['Log_ratio'] = df['Ratio'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    df_out = pd.DataFrame()
    for sex in df['Sex'].unique():
        for age_id in df['Age_Group'].unique():
            df_subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_id)].copy()
            if len(df_subset) == 0:
                continue
            
            lower_bound = df_subset['Log_ratio'].quantile(0.2)
            upper_bound = df_subset['Log_ratio'].quantile(0.80)
            
            median = df_subset['Log_ratio'].median()
            mad = (df_subset['Log_ratio'] - median).abs().median()
            df_subset['Deviation'] = df_subset['Log_ratio'].apply(
                lambda x: modified_zscore(x, median, mad)
            )
            
            df_subset['Outlier'] = (
                (df_subset['Log_ratio'] < lower_bound) | 
                (df_subset['Log_ratio'] > upper_bound)
            )
            
            df_out = pd.concat([df_out, df_subset], ignore_index=True)
            
    return df_out


def select_top_outliers(
    df_outliers: pd.DataFrame, 
    n_high: int = 20, 
    n_low: int = 10
) -> pd.DataFrame:
    """
    Select top N high and low degree outliers per sex-age group.
    
    This filters the outlier dataset to the most extreme cases for reporting.
    
    Args:
        df_outliers: DataFrame from detect_outliers_exact()
        n_high: Number of top high-degree outliers to select per group
        n_low: Number of top low-degree outliers to select per group
        
    Returns:
        DataFrame with top outliers, sorted by Sex, Age_Group, outlier_type, Log_ratio
        
    Examples:
        >>> df_outliers = detect_outliers_exact(df_all)
        >>> top_outliers = select_top_outliers(df_outliers, n_high=20, n_low=10)
    """
    logger.info(f"Selecting top {n_high} high and top {n_low} low outliers...")
    results = []
    
    for sex in ['Female', 'Male']:
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        for age_range in sorted(sex_data['Age_Range'].unique()):
            age_data = sex_data[sex_data['Age_Range'] == age_range]
            if len(age_data) == 0:
                continue
            
            age_median = age_data['Log_ratio'].median()
            
            high_degree = age_data[age_data['Log_ratio'] > age_median].nlargest(
                n_high, 'Log_ratio'
            ).copy()
            high_degree['outlier_type'] = 'high_degree'
            
            low_degree = age_data[age_data['Log_ratio'] <= age_median].nsmallest(
                n_low, 'Log_ratio'
            ).copy()
            low_degree['outlier_type'] = 'low_degree'
            
            results.append(high_degree)
            results.append(low_degree)
    
    table_data = pd.concat(results, ignore_index=True)
    table_data['age_num'] = table_data['Age_Group']
    table_data['type_order'] = table_data['outlier_type'].map({
        'high_degree': 0, 'low_degree': 1
    })
    
    return table_data.sort_values(
        ['Sex', 'age_num', 'type_order', 'Log_ratio'], 
        ascending=[True, True, True, False]
    )


def identify_high_mortality_sinks_zscore(
    df_all: pd.DataFrame, 
    top_percent: int = 20
) -> pd.DataFrame:
    """
    Identify high-mortality sinks using Z-score product method.
    
    High-mortality sinks are diseases that are:
    1. Central in the network (high betweenness centrality)
    2. Associated with high mortality rates
    
    These nodes are particularly harmful as they:
    - Lie on many shortest paths between other diseases (high betweenness)
    - Have high mortality rates
    - Represent critical intervention points in the comorbidity network
    
    Methodology:
    1. Calculate z-scores for betweenness and mortality within each sex-age group
    2. Compute z_product = z_betweenness × z_mortality
    3. Filter to nodes with BOTH positive z-scores (above average in both dimensions)
    4. Select top X% by z_product percentile
    5. Calculate geometric mean for ranking: sqrt(z_betweenness × z_mortality)
    
    Args:
        df_all: DataFrame with columns: Sex, Age_Group, ICD_Code, Betweenness, Mortality
        top_percent: Percentile threshold (default: 20 for top 20%)
        
    Returns:
        DataFrame with high-mortality sinks including z-scores and rankings
        
    Paper Reference:
        Section on "High-mortality sinks" - nodes with high betweenness and mortality
        
    Examples:
        >>> df_all = load_all_network_data()
        >>> sinks = identify_high_mortality_sinks_zscore(df_all, top_percent=20)
    """
    logger.info(f"Identifying high-mortality sinks (Top {top_percent}%)...")
    all_high = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & (df_all['Age_Group'] == age_group)
            ].copy()
            if len(subset) == 0:
                continue
            
            # Compute z-scores for betweenness and mortality
            subset['z_betweenness'] = compute_z_score(subset['Betweenness'])
            subset['z_mortality'] = compute_z_score(subset['Mortality'])
            
            # Compute product and geometric mean
            subset['z_product'] = subset['z_betweenness'] * subset['z_mortality']
            subset['z_geom_mean'] = np.where(
                (subset['z_betweenness'] > 0) & (subset['z_mortality'] > 0),
                np.sqrt(subset['z_betweenness'] * subset['z_mortality']),
                0
            )
            
            # Select top percentile with both positive z-scores
            threshold = subset['z_product'].quantile((100 - top_percent) / 100)
            high_nodes = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mortality'] > 0) &
                (subset['z_product'] >= threshold)
            ].copy()
            
            if len(high_nodes) > 0:
                all_high.append(high_nodes)
                
    return pd.concat(all_high, ignore_index=True) if all_high else pd.DataFrame()


@app.command()
def run_outlier_detection(output_filename: str = "outliers_data_S1.csv"):
    """
    Main entry point for outlier detection (Supplementary Table S1).
    
    This generates a table of high and low degree outliers for all sex-age combinations.
    """
    logger.info("Starting Standalone Outlier Detection")
    all_data = []

    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age Group {age_id}...")
            df = NetworkAnalyzer.load_node_metrics(gender, age_id)
            if not df.empty and 'Prevalence' in df.columns:
                # Filter for valid prevalence > 0
                df = df[df['Prevalence'] > 0]
                if not df.empty:
                    all_data.append(df)

    if not all_data:
        logger.error("No valid data loaded.")
        raise typer.Exit(code=1)

    df_all = pd.concat(all_data, ignore_index=True)
    
    df_outliers = detect_outliers_exact(df_all)
    df_outliers = NetworkAnalyzer.add_english_descriptions(df_outliers)
    df_final = select_top_outliers(df_outliers)
    
    output_path = PROCESSED_DATA_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logger.success(f"Saved to: {output_path}")


@app.command()
def run_mortality_sinks(
    output_filename: str = "high_mortality_sinks_ZSCORE.csv", 
    top_percent: int = 20
):
    """
    Main entry point for high-mortality sinks analysis.
    
    This generates a table of diseases with high betweenness and mortality.
    """
    logger.info("Starting Mortality Sinks Analysis...")
    all_data = []
    
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age {age_id}...")
            df = NetworkAnalyzer.load_node_metrics(gender, age_id)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    df_high = identify_high_mortality_sinks_zscore(df_all, top_percent=top_percent)
    if df_high.empty:
        logger.warning("No sinks found.")
        return

    df_high = NetworkAnalyzer.add_english_descriptions(df_high)
    out_csv = PROCESSED_DATA_DIR / output_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_high.to_csv(out_csv, index=False)
    logger.success(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    app()
