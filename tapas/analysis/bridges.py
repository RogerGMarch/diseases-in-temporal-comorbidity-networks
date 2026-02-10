"""
Critical bridge edges analysis.

This module identifies edges (disease pairs) with:
1. High edge betweenness centrality (critical connections)
2. High mortality difference between connected diseases

Paper Reference:
- "High-mortality bridges" - edges connecting diseases with disparate mortality rates
"""

import pandas as pd
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
    
    Critical bridges are disease pairs (edges) that:
    1. Connect different parts of the network (high edge betweenness)
    2. Have large mortality differences between the diseases
    
    These represent important transitions in the disease network where
    patients move between diseases with very different mortality rates.
    
    Methodology:
    1. Calculate z-scores for edge betweenness and mortality difference
    2. Compute z_product = z_betweenness × z_mortality_diff
    3. Filter to edges with BOTH positive z-scores
    4. Apply minimum mortality difference threshold (default: 30%)
    5. Select top X% by z_product percentile
    
    Args:
        df_all: DataFrame with columns: Sex, Age_Group, ICD_Code_1, ICD_Code_2,
                Edge_Betweenness, Mortality_1, Mortality_2, Mortality_Diff
        top_percent: Percentile threshold (default: 5 for top 5%)
        min_mort_diff: Minimum absolute mortality difference (default: 0.30)
        
    Returns:
        DataFrame with critical bridge edges including z-scores
        
    Examples:
        >>> df_all = load_all_edge_data()
        >>> bridges = identify_critical_bridges(df_all, top_percent=5, min_mort_diff=0.30)
    """
    logger.info(
        f"Identifying bridge edges (Top {top_percent}%, Min Diff > {min_mort_diff})..."
    )
    all_bridges = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[
                (df_all['Sex'] == sex) & (df_all['Age_Group'] == age_group)
            ].copy()
            if len(subset) == 0:
                continue
            
            # Compute z-scores
            subset['z_betweenness'] = compute_z_score(subset['Edge_Betweenness'])
            subset['z_mort_diff'] = compute_z_score(subset['Mortality_Diff'])
            
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            threshold = subset['z_product'].quantile((100 - top_percent) / 100)
            
            # Select bridges with both positive z-scores and sufficient mortality diff
            bridges = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mort_diff'] > 0) &
                (subset['z_product'] >= threshold) &
                (subset['Mortality_Diff'] >= min_mort_diff)
            ].copy()
            
            if len(bridges) > 0:
                all_bridges.append(bridges)
                
    return pd.concat(all_bridges, ignore_index=True) if all_bridges else pd.DataFrame()


@app.command()
def main(
    output_filename: str = "bridge_edges_mortality_ZSCORE.csv",
    top_percent: float = 5,
    min_mort_diff: float = 0.30
):
    """
    Main entry point for critical bridge edges analysis.
    
    This generates a table of critical bridge edges for all sex-age combinations.
    """
    logger.info("Starting Bridge Edges Analysis...")
    all_data = []
    
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age {age_id}...")
            df = NetworkAnalyzer.load_edge_metrics(gender, age_id)
            if not df.empty:
                all_data.append(df)
            
    if not all_data:
        raise typer.Exit(code=1)
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    df_bridges = identify_critical_bridges(df_all, top_percent, min_mort_diff)
    if df_bridges.empty:
        return

    df_bridges = NetworkAnalyzer.add_english_descriptions(df_bridges)
    
    out_csv = PROCESSED_DATA_DIR / output_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bridges.to_csv(out_csv, index=False)
    logger.success(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    app()
