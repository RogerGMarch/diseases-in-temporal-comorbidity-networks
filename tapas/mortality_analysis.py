import pandas as pd
import numpy as np
import typer
from loguru import logger
from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def identify_high_mortality_sinks_zscore(df_all: pd.DataFrame, top_percent: int = 20) -> pd.DataFrame:
    """Identify high-mortality sinks using Z-score product method."""
    logger.info(f"Identifying high-mortality sinks (Top {top_percent}%)...")
    all_high = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[(df_all['Sex'] == sex) & (df_all['Age_Group'] == age_group)].copy()
            if len(subset) == 0: continue
            
            for col in ['Betweenness', 'Mortality']:
                mean, std = subset[col].mean(), subset[col].std()
                subset[f'z_{col.lower()}'] = (subset[col] - mean) / std if std > 0 else 0
            
            subset['z_product'] = subset['z_betweenness'] * subset['z_mortality']
            subset['z_geom_mean'] = np.where(
                (subset['z_betweenness'] > 0) & (subset['z_mortality'] > 0),
                np.sqrt(subset['z_betweenness'] * subset['z_mortality']), 0
            )
            
            threshold = subset['z_product'].quantile((100 - top_percent) / 100)
            high_nodes = subset[
                (subset['z_betweenness'] > 0) &
                (subset['z_mortality'] > 0) &
                (subset['z_product'] >= threshold)
            ].copy()
            
            if len(high_nodes) > 0: all_high.append(high_nodes)
                
    return pd.concat(all_high, ignore_index=True) if all_high else pd.DataFrame()

@app.command()
def main(output_filename: str = "high_mortality_sinks_ZSCORE.csv", top_percent: int = 20):
    """Generate High Mortality Sinks table."""
    logger.info("Starting Mortality Sinks Analysis...")
    all_data = []
    
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age {age_id}...")
            # Use new centralized loader
            df = NetworkAnalyzer.load_node_metrics(gender, age_id)
            if not df.empty: all_data.append(df)
    
    if not all_data: raise typer.Exit(code=1)
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