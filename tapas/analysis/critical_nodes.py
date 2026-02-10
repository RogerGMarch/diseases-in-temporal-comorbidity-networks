"""
Critical nodes intersection analysis.

This module identifies the intersection of:
1. High-degree outliers (diseases with unusually high connections)
2. High-mortality sinks (diseases with high betweenness and mortality)

These represent the most critical diseases in the comorbidity network -
both highly connected and associated with high mortality.

Paper Reference:
- Critical disease nodes - intersection of structural importance and clinical severity
"""

import pandas as pd
import typer
from loguru import logger

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer
from tapas.analysis.outliers import (
    detect_outliers_exact,
    identify_high_mortality_sinks_zscore
)

app = typer.Typer()


def find_intersection(
    df_outliers: pd.DataFrame,
    df_sinks: pd.DataFrame
) -> pd.DataFrame:
    """
    Find intersection of high-degree outliers and high-mortality sinks.
    
    Creates a unique node identifier and finds diseases that appear in both datasets.
    
    Args:
        df_outliers: DataFrame from detect_outliers_exact()
        df_sinks: DataFrame from identify_high_mortality_sinks_zscore()
        
    Returns:
        DataFrame with intersection, merging metrics from both analyses
        
    Examples:
        >>> outliers = detect_outliers_exact(df_all)
        >>> sinks = identify_high_mortality_sinks_zscore(df_all)
        >>> critical = find_intersection(outliers, sinks)
    """
    df_outliers = df_outliers.copy()
    
    # Create unique node identifiers
    df_outliers['node_id'] = (
        df_outliers['Sex'] + '_' + 
        df_outliers['Age_Group'].astype(str) + '_' + 
        df_outliers['ICD_Code']
    )
    df_sinks['node_id'] = (
        df_sinks['Sex'] + '_' + 
        df_sinks['Age_Group'].astype(str) + '_' + 
        df_sinks['ICD_Code']
    )
    
    intersection_ids = set(df_outliers['node_id']) & set(df_sinks['node_id'])
    if not intersection_ids:
        return pd.DataFrame()
    
    # Merge on intersection
    df_int = df_sinks[df_sinks['node_id'].isin(intersection_ids)].copy()
    outlier_cols = df_outliers[['node_id', 'Log_ratio', 'Prevalence']].rename(
        columns={'Log_ratio': 'Log_Ratio'}
    )
    return df_int.merge(outlier_cols, on='node_id', how='left')


@app.command()
def main(top_percent_sinks: int = 40):
    """
    Complete pipeline for Critical Nodes intersection analysis.
    
    Identifies diseases that are both:
    - High-degree outliers (unusually connected)
    - High-mortality sinks (central and deadly)
    
    Args:
        top_percent_sinks: Percentile threshold for mortality sinks (default: 40%)
    """
    logger.info("Starting Critical Nodes Pipeline...")
    
    # 1. Load Data
    all_data = []
    for g in SEXES:
        for a in AGE_GROUPS.keys():
            df = NetworkAnalyzer.load_node_metrics(g, a)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise typer.Exit(code=1)
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 2. Outliers (High Degree)
    df_outliers_proc = detect_outliers_exact(df_all)
    # Filter for actual outliers with positive deviation
    df_high_degree = df_outliers_proc[
        (df_outliers_proc['Outlier']) & (df_outliers_proc['Deviation'] > 0)
    ].copy()
    
    # 3. Sinks (High Mortality + Betweenness)
    df_sinks = identify_high_mortality_sinks_zscore(df_all, top_percent=top_percent_sinks)
    
    # 4. Intersection
    df_final = find_intersection(df_high_degree, df_sinks)
    if df_final.empty:
        logger.warning("No intersection found.")
        return

    df_final = NetworkAnalyzer.add_english_descriptions(df_final)
    out_path = PROCESSED_DATA_DIR / 'critical_nodes_intersection_ZSCORE.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    logger.success(f"Saved to {out_path}")


if __name__ == "__main__":
    app()
