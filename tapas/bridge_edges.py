import pandas as pd
import typer
from loguru import logger
from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def identify_critical_bridges(df_all: pd.DataFrame, top_percent: float = 5, min_mort_diff: float = 0.30) -> pd.DataFrame:
    """Identify bridge edges using Z-score product method."""
    logger.info(f"Identifying bridge edges (Top {top_percent}%, Min Diff > {min_mort_diff})...")
    all_bridges = []
    
    for sex in df_all['Sex'].unique():
        for age_group in df_all['Age_Group'].unique():
            subset = df_all[(df_all['Sex'] == sex) & (df_all['Age_Group'] == age_group)].copy()
            if len(subset) == 0: continue
            
            for col in ['Edge_Betweenness', 'Mortality_Diff']:
                mean, std = subset[col].mean(), subset[col].std()
                col_name = 'z_betweenness' if col == 'Edge_Betweenness' else 'z_mort_diff'
                subset[col_name] = (subset[col] - mean) / std if std > 0 else 0
            
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            threshold = subset['z_product'].quantile((100 - top_percent) / 100)
            
            bridges = subset[
                (subset['z_betweenness'] > 0) & (subset['z_mort_diff'] > 0) &
                (subset['z_product'] >= threshold) & (subset['Mortality_Diff'] >= min_mort_diff)
            ].copy()
            
            if len(bridges) > 0: all_bridges.append(bridges)
                
    return pd.concat(all_bridges, ignore_index=True) if all_bridges else pd.DataFrame()

@app.command()
def main(output_filename: str = "bridge_edges_mortality_ZSCORE.csv", top_percent: float = 5, min_mort_diff: float = 0.30):
    """Generate Critical Bridge Edges table."""
    logger.info("Starting Bridge Edges Analysis...")
    all_data = []
    
    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age {age_id}...")
            # Use new centralized edge loader
            df = NetworkAnalyzer.load_edge_metrics(gender, age_id)
            if not df.empty: all_data.append(df)
            
    if not all_data: raise typer.Exit(code=1)
    df_all = pd.concat(all_data, ignore_index=True)
    
    df_bridges = identify_critical_bridges(df_all, top_percent, min_mort_diff)
    if df_bridges.empty: return

    df_bridges = NetworkAnalyzer.add_english_descriptions(df_bridges)
    
    out_csv = PROCESSED_DATA_DIR / output_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bridges.to_csv(out_csv, index=False)
    logger.success(f"Saved CSV to {out_csv}")

if __name__ == "__main__":
    app()