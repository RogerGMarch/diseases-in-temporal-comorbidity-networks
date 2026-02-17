"""
Outlier detection for comorbidity networks.

This module consolidates outlier analysis functionality including:
1. Degree-prevalence outlier detection (high/low degree relative to prevalence)
2. High-mortality sinks identification (high betweenness + high mortality)
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer
from tapas.utils.statistics import modified_zscore, compute_z_score, compute_log_ratio

app = typer.Typer()

def detect_outliers_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using 20th/80th percentile thresholds on Log(Degree/Prevalence).
    """
    # Safety check for required columns
    req_cols = ['Degree', 'Prevalence', 'Sex', 'Age_Group']
    if not all(col in df.columns for col in req_cols):
        raise KeyError(f"Missing columns for outlier detection. Found: {df.columns}")

    df = df.copy()
    
    # Calculate Log Ratio using robust utility
    df['Log_Ratio'] = compute_log_ratio(df['Degree'], df['Prevalence'])
    
    # Keep linear Ratio for reference/debugging if needed
    # Using numpy where to avoid ZeroDivisionError safely
    df['Ratio'] = np.where(df['Prevalence'] > 0, df['Degree'] / df['Prevalence'], 0.0)

    # Calculate Modified Z-Score per group (Robustness)
    # This identifies how far a node is from the group's trend
    # transform(modified_zscore) works because modified_zscore accepts the Series
    df['Deviation'] = df.groupby(['Sex', 'Age_Group'])['Log_Ratio'].transform(modified_zscore)

    # Determine Percentile Thresholds per group
    def get_outliers(group):
        valid_logs = group['Log_Ratio'].dropna()
        if valid_logs.empty:
            group['Outlier'] = False
            return group
            
        upper = valid_logs.quantile(0.80)
        lower = valid_logs.quantile(0.20)
        
        # Mark outliers
        group['Outlier'] = (group['Log_Ratio'] > upper) | (group['Log_Ratio'] < lower)
        return group

    df = df.groupby(['Sex', 'Age_Group'], group_keys=False).apply(get_outliers)
    
    return df

def identify_high_mortality_sinks_zscore(df: pd.DataFrame, top_percent: float = 20) -> pd.DataFrame:
    """
    Identify high-mortality sinks using Z-score product method.
    Nodes with High Betweenness AND High Mortality.
    """
    req_cols = ['Betweenness', 'Mortality_Rate', 'Sex', 'Age_Group']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logger.warning(f"Columns missing for sink analysis: {missing}")
        return pd.DataFrame()

    df = df.copy()

    # Calculate Z-scores within each Sex/Age group
    df['Z_Betweenness'] = df.groupby(['Sex', 'Age_Group'])['Betweenness'].transform(compute_z_score)
    df['Z_Mortality'] = df.groupby(['Sex', 'Age_Group'])['Mortality_Rate'].transform(compute_z_score)

    # Calculate Product
    df['Z_Score_Product'] = df['Z_Betweenness'] * df['Z_Mortality']

    # Filter: Must be high in BOTH (Positive Z-scores)
    candidates = df[(df['Z_Betweenness'] > 0) & (df['Z_Mortality'] > 0)].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Rank by Z-Score Product
    threshold = np.percentile(candidates['Z_Score_Product'], 100 - top_percent)
    sinks = candidates[candidates['Z_Score_Product'] >= threshold].copy()
    
    return sinks.sort_values('Z_Score_Product', ascending=False)

@app.command()
def main(
    output_filename: str = "outliers_data_S1.csv",
    sinks_filename: str = "high_mortality_sinks_ZSCORE.csv"
):
    """
    Run full outlier and sink analysis.
    """
    logger.info("Starting Outlier & Sink Analysis...")
    
    all_data = []
    
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            df = NetworkAnalyzer.load_node_metrics(gender, age_id)
            if not df.empty:
                df['Sex'] = gender
                df['Age_Group'] = age_id
                all_data.append(df)
    
    if not all_data:
        logger.error("No data found.")
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df_all)} nodes.")

    # Detect Degree Outliers
    logger.info("Detecting degree outliers...")
    df_outliers = detect_outliers_exact(df_all)
    
    df_outliers_save = df_outliers[df_outliers['Outlier']].copy()
    df_outliers_save = NetworkAnalyzer.add_english_descriptions(df_outliers_save)
    
    out_path = PROCESSED_DATA_DIR / output_filename
    df_outliers_save.to_csv(out_path, index=False)
    logger.success(f"Saved outliers to {out_path}")

    # Identify Sinks
    logger.info("Identifying high-mortality sinks...")
    df_sinks = identify_high_mortality_sinks_zscore(df_all)
    
    if not df_sinks.empty:
        df_sinks = NetworkAnalyzer.add_english_descriptions(df_sinks)
        sinks_path = PROCESSED_DATA_DIR / sinks_filename
        df_sinks.to_csv(sinks_path, index=False)
        logger.success(f"Saved sinks to {sinks_path}")

if __name__ == "__main__":
    app()