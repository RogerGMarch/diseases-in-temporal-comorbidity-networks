"""
Summary Table Generation (Table 2).

Replicates '007_Table2_Summary.ipynb' to generate the final setup summary:
1. Counts High Degree Outliers using the 20th/80th percentile method (EXACT setup).
   - Forces 16 rows (matches notebook explicit loop).
2. Loads High-Mortality Sinks (Z-score top 20%) from processed data.
   - Only lists existing groups (matches notebook groupby behavior).
3. Loads High-Mortality Bridges (Z-score top 5%) from processed data.
   - Only lists existing groups (matches notebook groupby behavior).
4. Aggregates counts and exports to CSV.
"""

import pandas as pd
import typer
from loguru import logger
from pathlib import Path

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
# Reuse the exact outlier logic we implemented (inclusive 20th/80th percentile)
from tapas.analysis.outliers import get_all_outliers_exact

app = typer.Typer()

def count_high_degree_outliers() -> pd.DataFrame:
    """
    Count high degree outliers using the exact 20th/80th percentile method.
    """
    logger.info("1. Counting high degree outliers (80th percentile)...")
    
    # Load all data with exact 20/80 percentile detection
    df_outliers = get_all_outliers_exact()
    
    if df_outliers.empty:
        logger.warning("No outlier data returned.")
        return pd.DataFrame(columns=['Sex', 'Age_Group', 'Count'])

    # Filter for High Degree Outliers:
    # 1. Marked as Outlier (True)
    # 2. Positive Deviation (Upper tail / High Degree)
    high_degree = df_outliers[
        (df_outliers['Outlier'] == True) & 
        (df_outliers['Deviation'] > 0)
    ]
    
    # Group by Sex and Age_Group
    counts = high_degree.groupby(['Sex', 'Age_Group']).size().reset_index(name='Count')
    
    # MATCH NOTEBOOK BEHAVIOR:
    # The notebook explicitly iterates "for age_group in range(1, 9)", 
    # so it ALWAYS produces 16 rows for outliers, even if count is 0.
    full_index = pd.MultiIndex.from_product([SEXES, AGE_GROUPS.keys()], names=['Sex', 'Age_Group'])
    counts = counts.set_index(['Sex', 'Age_Group']).reindex(full_index, fill_value=0).reset_index()
    
    logger.info(f"   Total High Degree Outliers: {counts['Count'].sum()}")
    return counts

def load_counts_from_csv(filename: str, label: str) -> pd.DataFrame:
    """
    Generic helper to load a processed CSV file and count rows per Sex/Age group.
    
    MATCHING NOTEBOOK BEHAVIOR:
    The notebook uses `groupby`, which naturally omits groups with 0 counts.
    We DO NOT reindex here, so missing groups remain missing.
    """
    file_path = PROCESSED_DATA_DIR / filename
    logger.info(f"Loading {label} from {file_path.name}...")
    
    if not file_path.exists():
        logger.warning(f"   [!] File not found: {file_path}")
        return pd.DataFrame(columns=['Sex', 'Age_Group', 'Count'])

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning("   File is empty.")
            counts = pd.DataFrame(columns=['Sex', 'Age_Group', 'Count'])
        else:
            # Simple groupby matches notebook behavior (omits zeros)
            counts = df.groupby(['Sex', 'Age_Group']).size().reset_index(name='Count')
            
        logger.info(f"   Total: {counts['Count'].sum()}")
        return counts
        
    except Exception as e:
        logger.error(f"   Error reading file: {e}")
        return pd.DataFrame(columns=['Sex', 'Age_Group', 'Count'])

@app.command()
def main():
    """Main execution for Summary Table generation."""
    logger.info("="*60)
    logger.info("SUMMARY TABLE - FINAL SETUP")
    logger.info("="*60)
    
    # 1. High Degree Outliers (Calculated dynamically)
    # Should yield exactly 16 rows (2 sexes * 8 ages)
    df_outliers = count_high_degree_outliers()
    df_outliers['Type'] = 'High degree outliers (80th p)'
    
    # 2. High-Mortality Sinks (Loaded from CSV)
    # Should yield only existing rows (e.g., < 16)
    df_sinks = load_counts_from_csv('high_mortality_sinks_ZSCORE.csv', 'High-mortality sinks')
    df_sinks['Type'] = 'High-mortality sinks (Z-score 20th p)'
    
    # 3. High-Mortality Bridges (Loaded from CSV)
    # Should yield only existing rows (e.g., < 16)
    df_bridges = load_counts_from_csv('bridge_edges_mortality_ZSCORE.csv', 'High-mortality bridges')
    df_bridges['Type'] = 'High-mortality bridges (Z-score 5th p)'
    
    # Combine
    # Total rows should now match notebook (approx 39)
    df_final = pd.concat([df_outliers, df_sinks, df_bridges], ignore_index=True)
    
    # 4. Generate Output
    logger.info("Generating output...")
    
    # Save CSV
    csv_file = PROCESSED_DATA_DIR / 'summary_table_FINAL_SETUP.csv'
    df_final.to_csv(csv_file, index=False)
    logger.success(f"✓ Summary CSV saved to: {csv_file}")
    
    # Print Final Summary to Console
    logger.info("\nFINAL SUMMARY:")
    logger.info(f"Total Rows Generated: {len(df_final)}")
    
    for t in df_final['Type'].unique():
        subset = df_final[df_final['Type'] == t]
        total = subset['Count'].sum()
        female = subset[subset['Sex'] == 'Female']['Count'].sum()
        male = subset[subset['Sex'] == 'Male']['Count'].sum()
        logger.info(f"{t}: Total {total} (F: {female}, M: {male})")

if __name__ == "__main__":
    app()