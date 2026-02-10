import pandas as pd
import typer
from loguru import logger
from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS, INTERIM_DATA_DIR, DATA_DIR
from tapas.features import NetworkAnalyzer
from outlier_detection import detect_outliers_exact
from mortality_analysis import identify_high_mortality_sinks_zscore

app = typer.Typer()

def find_intersection(df_outliers: pd.DataFrame, df_sinks: pd.DataFrame) -> pd.DataFrame:
    """Find intersection based on custom node_id."""
    df_outliers = df_outliers.copy()
    # Ensure ID consistency
    df_outliers['node_id'] = df_outliers['Sex'] + '_' + df_outliers['Age_Group'].astype(str) + '_' + df_outliers['ICD_Code']
    df_sinks['node_id'] = df_sinks['Sex'] + '_' + df_sinks['Age_Group'].astype(str) + '_' + df_sinks['ICD_Code']
    
    intersection_ids = set(df_outliers['node_id']) & set(df_sinks['node_id'])
    if not intersection_ids: return pd.DataFrame()
    
    df_int = df_sinks[df_sinks['node_id'].isin(intersection_ids)].copy()
    outlier_cols = df_outliers[['node_id', 'Log_ratio', 'Prevalence']].rename(columns={'Log_ratio': 'Log_Ratio'})
    return df_int.merge(outlier_cols, on='node_id', how='left')

@app.command()
def main(top_percent_sinks: int = 40):
    """Complete pipeline for Critical Nodes."""
    logger.info("Starting Critical Nodes Pipeline...")
    
    # 1. Load Data
    all_data = []
    for g in SEXES:
        for a in AGE_GROUPS.keys():
            df = NetworkAnalyzer.load_node_metrics(g, a)
            if not df.empty: all_data.append(df)
    
    if not all_data: raise typer.Exit(code=1)
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 2. Outliers (Degree)
    df_outliers_proc = detect_outliers_exact(df_all)
    # Filter for actual outliers (positive deviation per notebook logic)
    df_high_degree = df_outliers_proc[(df_outliers_proc['Outlier']) & (df_outliers_proc['Deviation'] > 0)].copy()
    
    # 3. Sinks (Mortality)
    df_sinks = identify_high_mortality_sinks_zscore(df_all, top_percent=top_percent_sinks)
    
    # 4. Intersection
    df_final = find_intersection(df_high_degree, df_sinks)
    if df_final.empty:
        logger.warning("No intersection found.")
        return

    df_final = NetworkAnalyzer.add_english_descriptions(df_final)
    out_path = PROCESSED_DATA_DIR / 'critical_nodes_intersection_ZSCORE.csv'
    df_final.to_csv(out_path, index=False)
    logger.success(f"Saved to {out_path}")

if __name__ == "__main__":
    app()