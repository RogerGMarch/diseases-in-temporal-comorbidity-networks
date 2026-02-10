import pandas as pd
import numpy as np
import typer
from loguru import logger
from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def modified_zscore(x, median, mad):
    """Modified z-score: 0.6745 * (x - median) / mad"""
    if mad == 0: return np.nan
    return 0.6745 * (x - median) / mad

def detect_outliers_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers using the exact method from the notebook (20th/80th percentiles)."""
    logger.info("Detecting outliers (20th/80th percentile)...")
    
    df['Ratio'] = df['Degree'] / df['Prevalence']
    df['Log_ratio'] = df['Ratio'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    df_out = pd.DataFrame()
    for sex in df['Sex'].unique():
        for age_id in df['Age_Group'].unique():
            df_subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_id)].copy()
            if len(df_subset) == 0: continue
            
            lower_bound = df_subset['Log_ratio'].quantile(0.2)
            upper_bound = df_subset['Log_ratio'].quantile(0.80)
            
            median = df_subset['Log_ratio'].median()
            mad = (df_subset['Log_ratio'] - median).abs().median()
            df_subset['Deviation'] = df_subset['Log_ratio'].apply(lambda x: modified_zscore(x, median, mad))
            
            df_subset['Outlier'] = (df_subset['Log_ratio'] < lower_bound) | (df_subset['Log_ratio'] > upper_bound)
            df_out = pd.concat([df_out, df_subset], ignore_index=True)
            
    return df_out

def select_top_outliers(df_outliers: pd.DataFrame, n_high: int = 20, n_low: int = 10) -> pd.DataFrame:
    """Select top N high and low degree outliers per sex-age group."""
    logger.info(f"Selecting top {n_high} high and top {n_low} low outliers...")
    results = []
    
    for sex in ['Female', 'Male']:
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        for age_range in sorted(sex_data['Age_Range'].unique()):
            age_data = sex_data[sex_data['Age_Range'] == age_range]
            if len(age_data) == 0: continue
            
            age_median = age_data['Log_ratio'].median()
            
            high_degree = age_data[age_data['Log_ratio'] > age_median].nlargest(n_high, 'Log_ratio').copy()
            high_degree['outlier_type'] = 'high_degree'
            
            low_degree = age_data[age_data['Log_ratio'] <= age_median].nsmallest(n_low, 'Log_ratio').copy()
            low_degree['outlier_type'] = 'low_degree'
            
            results.append(high_degree)
            results.append(low_degree)
    
    table_data = pd.concat(results, ignore_index=True)
    table_data['age_num'] = table_data['Age_Group'] # Assuming int passed through
    table_data['type_order'] = table_data['outlier_type'].map({'high_degree': 0, 'low_degree': 1})
    
    return table_data.sort_values(['Sex', 'age_num', 'type_order', 'Log_ratio'], ascending=[True, True, True, False])

@app.command()
def main(output_filename: str = "outliers_data_S1.csv"):
    """Main entry point for outlier detection."""
    logger.info("Starting Standalone Outlier Detection")
    all_data = []

    for gender in SEXES:
        for age_id in AGE_GROUPS.keys():
            logger.info(f"Processing {gender} - Age Group {age_id}...")
            # Use new centralized loader
            df = NetworkAnalyzer.load_node_metrics(gender, age_id)
            if not df.empty and 'Prevalence' in df.columns:
                # Filter for valid prevalence > 0 to match logic
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

if __name__ == "__main__":
    app()