"""
Standalone Outlier Detection and Table Generation.

This script implements the methodology from the original notebook to:
1. Load network and prevalence data across all demographics
2. Detect outliers using the 5th/95th percentile method on Log(Degree/Prevalence)
3. Select the top high-degree and low-degree outliers
4. Export the final dataset to CSV
"""

import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()


def detect_outliers(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using 5th and 95th percentile method.
    
    Calculates the Ratio (Degree/Prevalence) and Log_ratio.
    Outliers are defined as diseases falling outside the 5th-95th percentile
    range of the Log_ratio distribution within each Sex/Age group.
    """
    logger.info("Detecting outliers using 5th/95th percentile method...")
    
    # Calculate ratio and log ratio
    # Avoid division by zero
    df_all['Ratio'] = df_all.apply(
        lambda row: row['Degree'] / row['Prevalence'] if row['Prevalence'] > 0 else np.nan, 
        axis=1
    )
    
    # Calculate Log10 of ratio
    df_all['Log_ratio'] = df_all['Ratio'].apply(
        lambda x: np.log10(x) if pd.notnull(x) and x > 0 else np.nan
    )
    
    # Drop rows where Log_ratio could not be calculated
    df_clean = df_all.dropna(subset=['Log_ratio']).copy()
    
    all_outliers = []
    
    # Process each sex-age group separately
    for sex in SEXES:
        # iterate through age groups present in data
        for age_group_id in df_clean['Age_Group'].unique():
            subset = df_clean[
                (df_clean['Sex'] == sex) & 
                (df_clean['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate percentile thresholds
            lower_bound = subset['Log_ratio'].quantile(0.05)
            upper_bound = subset['Log_ratio'].quantile(0.95)
            
            # Mark outliers
            subset['Outlier'] = (
                (subset['Log_ratio'] < lower_bound) | 
                (subset['Log_ratio'] > upper_bound)
            )
            
            # Keep only outliers
            outliers = subset[subset['Outlier'] == True]
            if len(outliers) > 0:
                all_outliers.append(outliers)
    
    if not all_outliers:
        return pd.DataFrame()
        
    return pd.concat(all_outliers, ignore_index=True)


def select_top_outliers(df_outliers: pd.DataFrame, n_high: int = 20, n_low: int = 10) -> pd.DataFrame:
    """
    Select top N high and low degree outliers per sex-age group.
    
    Ranking is based on the magnitude of the Log_ratio relative to the group median.
    """
    logger.info(f"Selecting top {n_high} high and top {n_low} low outliers per group...")
    
    results = []
    
    for sex in SEXES:
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        
        # Sort age groups to ensure order
        for age_group_id in sorted(sex_data['Age_Group'].unique()):
            age_data = sex_data[sex_data['Age_Group'] == age_group_id]
            
            if len(age_data) == 0:
                continue
            
            # Split by median to get high/low differentiation within the outlier set
            # Note: In the notebook, median is calculated on the subset. 
            # High Log_ratio = Higher degree than expected from prevalence
            age_median = age_data['Log_ratio'].median()
            
            # Top N high degree outliers (Largest Log_ratios)
            high_degree = age_data[age_data['Log_ratio'] > age_median].nlargest(n_high, 'Log_ratio')
            high_degree = high_degree.copy()
            high_degree['outlier_type'] = 'high_degree'
            
            # Top N low degree outliers (Smallest Log_ratios)
            low_degree = age_data[age_data['Log_ratio'] <= age_median].nsmallest(n_low, 'Log_ratio')
            low_degree = low_degree.copy()
            low_degree['outlier_type'] = 'low_degree'
            
            results.append(high_degree)
            results.append(low_degree)
    
    if not results:
        return pd.DataFrame()

    # Combine and sort for final presentation
    table_data = pd.concat(results, ignore_index=True)
    
    # Create helper columns for sorting to match notebook output structure
    table_data['type_order'] = table_data['outlier_type'].map({
        'high_degree': 0, 'low_degree': 1
    })
    
    # Sort: Sex -> Age -> Type (High then Low) -> Log_ratio (descending)
    table_data = table_data.sort_values(
        ['Sex', 'Age_Group', 'type_order', 'Log_ratio'], 
        ascending=[True, True, True, False]
    )
    
    # Clean up helper column
    table_data = table_data.drop(columns=['type_order'])
    
    return table_data


@app.command()
def main():
    """Main execution function to generate outlier table."""
    logger.info("Starting Outlier Detection Pipeline")
    
    # Step 1: Load all data
    logger.info("Step 1: Loading network and prevalence data...")
    all_data = []
    
    # Iterate through all demographics defined in config
    for sex in SEXES:
        for age_id, age_range in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex} data"):
            # Use existing feature loader to get Node Degree + Prevalence
            # We don't apply a threshold here as per original notebook logic (weighted degree check)
            # but original notebook checks `degree > 0`. load_node_metrics does that.
            df = NetworkAnalyzer.load_node_metrics(sex, age_id)
            
            if not df.empty:
                # Filter for valid prevalence and degree
                # The notebook filters: degree > 0 and prevalence > 0
                df = df[(df['Degree'] > 0) & (df['Prevalence'] > 0)].copy()
                all_data.append(df)
            else:
                logger.warning(f"No data found for {sex} Age {age_id}")
    
    if not all_data:
        logger.error("No data loaded. Check data paths.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total diseases with degree > 0 and prevalence > 0: {len(df_all)}")
    
    # Step 2: Detect outliers
    logger.info("Step 2: Detecting outliers...")
    df_outliers = detect_outliers(df_all)
    logger.info(f"Total outliers detected: {len(df_outliers)}")
    
    # Step 3: Add English descriptions
    logger.info("Step 3: Adding English descriptions...")
    # Using the helper from NetworkAnalyzer
    df_outliers = NetworkAnalyzer.add_english_descriptions(df_outliers)
    
    # Step 4: Select top outliers for table
    logger.info("Step 4: Selecting top outliers for table...")
    table_data = select_top_outliers(df_outliers, n_high=20, n_low=10)
    logger.info(f"Rows in final table: {len(table_data)}")
    
    # Step 5: Save CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = PROCESSED_DATA_DIR / 'outliers_data_FINAL.csv'
    
    # Reorder columns to match logical flow
    cols = [
        'Sex', 'Age_Group', 'Age_Range', 'ICD_Code', 
        'Description_GER', 'Description_Eng', 
        'Degree', 'Prevalence', 'Ratio', 'Log_ratio', 
        'outlier_type', 'Betweenness', 'Mortality'
    ]
    # Only select columns that exist (Betweenness/Mortality might be missing if data wasn't found)
    cols = [c for c in cols if c in table_data.columns]
    
    table_data[cols].to_csv(csv_file, index=False)
    logger.success(f"✓ Data CSV saved to: {csv_file}")
    
    # Print summary statistics similar to notebook
    logger.info("\n" + "="*40)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*40)
    
    for sex in SEXES:
        sex_df = table_data[table_data['Sex'] == sex]
        high = len(sex_df[sex_df['outlier_type'] == 'high_degree'])
        low = len(sex_df[sex_df['outlier_type'] == 'low_degree'])
        logger.info(f"{sex}: {high} high + {low} low = {len(sex_df)} total")

if __name__ == '__main__':
    app()