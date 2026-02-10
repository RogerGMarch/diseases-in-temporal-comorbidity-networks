import pandas as pd
import numpy as np
import typer
from loguru import logger
from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()

@app.command()
def main():
    """Generate summary table from processed files."""
    logger.info("Generating Final Summary Table...")
    
    # 1. Count High Degree Outliers (Re-calculating logic from table 2 script)
    all_counts = []
    for g in SEXES:
        for a in AGE_GROUPS.keys():
            df = NetworkAnalyzer.load_node_metrics(g, a)
            count = 0
            if not df.empty and 'Prevalence' in df.columns:
                valid = df[(df['Degree'] > 0) & (df['Prevalence'] > 0)].copy()
                if not valid.empty:
                    valid['log_ratio'] = np.log10(valid['Degree'] / valid['Prevalence'])
                    upper = valid['log_ratio'].quantile(0.80)
                    count = len(valid[valid['log_ratio'] >= upper])
            all_counts.append({'Sex': g, 'Age_Group': a, 'Count': count})
    df_outliers = pd.DataFrame(all_counts)
    
    # 2. Load Sinks & Bridges
    try:
        df_sinks = pd.read_csv(PROCESSED_DATA_DIR / 'high_mortality_sinks_ZSCORE.csv')
        sinks_count = df_sinks.groupby(['Sex', 'Age_Group']).size().reset_index(name='Count')
        
        df_bridges = pd.read_csv(PROCESSED_DATA_DIR / 'bridge_edges_mortality_ZSCORE.csv')
        bridges_count = df_bridges.groupby(['Sex', 'Age_Group']).size().reset_index(name='Count')
    except FileNotFoundError as e:
        logger.error(f"Missing processed file: {e}. Run analysis scripts first.")
        raise typer.Exit(code=1)

    # 3. Combine
    df_outliers['Type'] = 'High degree outliers'
    sinks_count['Type'] = 'High-mortality sinks'
    bridges_count['Type'] = 'High-mortality bridges'
    
    df_final = pd.concat([df_outliers, sinks_count, bridges_count], ignore_index=True)
    df_final.to_csv(PROCESSED_DATA_DIR / 'summary_table_FINAL.csv', index=False)
    logger.success("Summary saved.")

if __name__ == "__main__":
    app()