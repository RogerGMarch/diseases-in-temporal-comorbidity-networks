"""
Critical nodes intersection analysis.
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
    """
    # Filter for High Degree Outliers (Positive deviation)
    if 'Outlier' in df_outliers.columns and 'Deviation' in df_outliers.columns:
        high_degree = df_outliers[
            (df_outliers['Outlier']) & (df_outliers['Deviation'] > 0)
        ].copy()
    else:
        logger.warning("Outlier dataframe missing required columns.")
        return pd.DataFrame()

    # Merge Keys
    merge_keys = ['Sex', 'Age_Group', 'ICD_Code']
    
    # Ensure Sinks have keys
    missing_keys = [k for k in merge_keys if k not in df_sinks.columns]
    if missing_keys:
         logger.warning(f"Sinks dataframe missing keys: {missing_keys}")
         return pd.DataFrame()

    intersection = pd.merge(
        high_degree,
        df_sinks,
        on=merge_keys,
        how='inner',
        suffixes=('_Outlier', '_Sink')
    )
    
    return intersection

@app.command()
def main(
    output_filename: str = "critical_nodes_intersection_ZSCORE.csv",
    top_percent_sinks: int = 20
):
    """
    Identify Critical Nodes (Intersection).
    """
    logger.info("Starting Critical Nodes Pipeline...")
    
    all_data = []
    
    # 1. LOAD DATA
    for g in SEXES:
        for a in AGE_GROUPS.keys():
            df = NetworkAnalyzer.load_node_metrics(g, a)
            if not df.empty:
                df['Sex'] = g
                df['Age_Group'] = a
                all_data.append(df)
    
    if not all_data:
        raise typer.Exit(code=1)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 2. Get Outliers
    logger.info("Calculating Outliers...")
    df_outliers_full = detect_outliers_exact(df_all)
    
    # 3. Get Sinks
    logger.info("Calculating Sinks...")
    df_sinks = identify_high_mortality_sinks_zscore(df_all, top_percent=top_percent_sinks)
    
    # 4. Find Intersection
    logger.info("Finding Intersection...")
    df_critical = find_intersection(df_outliers_full, df_sinks)
    
    if df_critical.empty:
        logger.warning("No critical nodes (intersection) found.")
        return

    # Add descriptions
    df_critical = NetworkAnalyzer.add_english_descriptions(df_critical)
    
    out_path = PROCESSED_DATA_DIR / output_filename
    df_critical.to_csv(out_path, index=False)
    logger.success(f"Saved {len(df_critical)} critical nodes to {out_path}")

if __name__ == "__main__":
    app()