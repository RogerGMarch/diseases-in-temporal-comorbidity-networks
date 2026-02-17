"""
Critical bridge edges analysis.

This module identifies edges (disease pairs) with:
1. High edge betweenness centrality (critical connections)
2. High mortality difference between connected diseases

Paper Reference:
- "High-mortality bridges" - edges connecting diseases with disparate mortality rates
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer
from tapas.utils.statistics import compute_z_score

app = typer.Typer()

def identify_critical_bridges(
    df_all: pd.DataFrame,
    top_percent: float = 5,
    min_mort_diff: float = 0.30
) -> pd.DataFrame:
    """
    Identify bridge edges using Z-score product method.
    """
    req_cols = ['Edge_Betweenness', 'Mortality_Diff', 'Sex', 'Age_Group']
    if not all(col in df_all.columns for col in req_cols):
         raise KeyError(f"Missing columns in edge data. Found: {df_all.columns}")

    df = df_all.copy()

    # Calculate Z-scores WITHIN groups (Sex/Age)
    # This compares an edge to other edges in the *same* network
    df['Z_Betweenness'] = df.groupby(['Sex', 'Age_Group'])['Edge_Betweenness'].transform(compute_z_score)
    df['Z_Mortality_Diff'] = df.groupby(['Sex', 'Age_Group'])['Mortality_Diff'].transform(compute_z_score)

    # Product of Z-scores
    df['Z_Score_Product'] = df['Z_Betweenness'] * df['Z_Mortality_Diff']

    # Filter
    # 1. Both metrics must be above average (positive Z)
    # 2. Mortality difference must be clinically significant (>= min_mort_diff)
    candidates = df[
        (df['Z_Betweenness'] > 0) & 
        (df['Z_Mortality_Diff'] > 0) & 
        (df['Mortality_Diff'] >= min_mort_diff)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Select top X percent based on the Product Score
    threshold = np.percentile(candidates['Z_Score_Product'], 100 - top_percent)
    bridges = candidates[candidates['Z_Score_Product'] >= threshold].copy()

    return bridges.sort_values('Z_Score_Product', ascending=False)


@app.command()
def main(
    output_filename: str = "bridge_edges_mortality_ZSCORE.csv",
    top_percent: float = 5,
    min_mort_diff: float = 0.30
):
    """
    Main entry point for critical bridge edges analysis.
    """
    logger.info("Starting Bridge Edges Analysis...")
    all_data = []
    
    # 1. LOAD DATA CORRECTLY
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age {age_id}...")
            # Note: ensure load_edge_metrics returns Edge_Betweenness and Mortality_Diff
            df = NetworkAnalyzer.load_edge_metrics(gender, age_id)
            if not df.empty:
                # --- FIX: Inject the group keys ---
                df['Sex'] = gender
                df['Age_Group'] = age_id
                all_data.append(df)
            
    if not all_data:
        logger.error("No edge data found.")
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df_all)} edges.")
    
    # 2. Identify Bridges
    df_bridges = identify_critical_bridges(df_all, top_percent, min_mort_diff)
    
    if df_bridges.empty:
        logger.warning("No critical bridges found matching criteria.")
        return

    # Add descriptions
    # Edge df usually has ICD_Code_1, ICD_Code_2. 
    # add_english_descriptions usually handles 'ICD_Code' column.
    # We might need to map manually if the helper doesn't support dual codes.
    # Assuming NetworkAnalyzer has a helper or we skip for now.
    
    out_csv = PROCESSED_DATA_DIR / output_filename
    df_bridges.to_csv(out_csv, index=False)
    logger.success(f"Saved {len(df_bridges)} bridges to {out_csv}")

if __name__ == "__main__":
    app()