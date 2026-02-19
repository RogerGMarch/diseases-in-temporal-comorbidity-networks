"""
Standalone Outlier Detection and Table Generation.

This script implements the methodology from the original notebook to:
1. Load network and prevalence data across all demographics
2. Detect outliers using the percentiles method on Log(Degree/Prevalence)
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


def modified_zscore(x, median, mad):
    """
    Calculate Modified Z-score (exact notebook implementation).
    Factor 0.6745 makes the MAD consistent with the standard deviation for normal distributions.
    """
    if mad == 0:
        return 0
    return 0.6745 * (x - median) / mad


def detect_outliers(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Standard outlier detection using 5th and 95th percentiles (exclusive).
    Kept for backward compatibility.
    """
    return _detect_outliers_internal(df_all, lower_q=0.05, upper_q=0.95, inclusive=False)


def detect_outliers_exact(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Exact notebook replication using 20th and 80th percentiles.
    
    CRITICAL CHANGE: Uses inclusive thresholds (>= and <=) to match 
    Notebook 007's logic, resulting in 786 high-degree outliers instead of 778.
    """
    # Notebook 007 uses >= for the upper bound, so we must use inclusive=True
    return _detect_outliers_internal(df_all, lower_q=0.20, upper_q=0.80, inclusive=True)


def _detect_outliers_internal(df_all: pd.DataFrame, lower_q: float, upper_q: float, inclusive: bool = False) -> pd.DataFrame:
    """
    Internal shared logic for outlier detection with customizable quantiles and inclusivity.
    """
    logger.info(f"Detecting outliers ({'inclusive' if inclusive else 'strict'}) using percentiles: {lower_q*100:.0f}th / {upper_q*100:.0f}th...")
    
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
        for age_group_id in df_clean['Age_Group'].unique():
            subset = df_clean[
                (df_clean['Sex'] == sex) & 
                (df_clean['Age_Group'] == age_group_id)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # 1. Percentile Thresholds
            lower_bound = subset['Log_ratio'].quantile(lower_q)
            upper_bound = subset['Log_ratio'].quantile(upper_q)
            
            # 2. Modified Z-score (Deviation)
            median = subset['Log_ratio'].median()
            mad = (subset['Log_ratio'] - median).abs().median()
            
            subset['Deviation'] = subset['Log_ratio'].apply(
                lambda x: modified_zscore(x, median, mad)
            )
            
            # 3. Mark outliers
            if inclusive:
                # Matches Notebook 007 (>= 80th percentile)
                subset['Outlier'] = (
                    (subset['Log_ratio'] <= lower_bound) | 
                    (subset['Log_ratio'] >= upper_bound)
                )
            else:
                # Matches Notebook 006 strict logic (> 80th percentile)
                subset['Outlier'] = (
                    (subset['Log_ratio'] < lower_bound) | 
                    (subset['Log_ratio'] > upper_bound)
                )
            
            all_outliers.append(subset)
    
    if not all_outliers:
        return pd.DataFrame()
        
    return pd.concat(all_outliers, ignore_index=True)


def select_top_outliers(df_all: pd.DataFrame, n_high: int = 20, n_low: int = 10) -> pd.DataFrame:
    """
    Select top N high and low degree outliers per sex-age group.
    Only considers rows where Outlier == True.
    """
    logger.info(f"Selecting top {n_high} high and top {n_low} low outliers per group...")
    
    df_outliers = df_all[df_all['Outlier'] == True].copy()
    results = []
    
    for sex in SEXES:
        sex_data = df_outliers[df_outliers['Sex'] == sex]
        
        for age_group_id in sorted(sex_data['Age_Group'].unique()):
            age_data = sex_data[sex_data['Age_Group'] == age_group_id]
            
            if len(age_data) == 0:
                continue
            
            age_median = age_data['Log_ratio'].median()
            
            # Note: For top N selection, we just take the largest values, 
            # so strict/inclusive inequality doesn't affect the *ordering*, just the pool size.
            high_degree = age_data[age_data['Log_ratio'] > age_median].nlargest(n_high, 'Log_ratio').copy()
            high_degree['outlier_type'] = 'high_degree'
            
            low_degree = age_data[age_data['Log_ratio'] <= age_median].nsmallest(n_low, 'Log_ratio').copy()
            low_degree['outlier_type'] = 'low_degree'
            
            results.append(high_degree)
            results.append(low_degree)
    
    if not results:
        return pd.DataFrame()

    table_data = pd.concat(results, ignore_index=True)
    table_data['type_order'] = table_data['outlier_type'].map({'high_degree': 0, 'low_degree': 1})
    
    table_data = table_data.sort_values(
        ['Sex', 'Age_Group', 'type_order', 'Log_ratio'], 
        ascending=[True, True, True, False]
    )
    
    return table_data.drop(columns=['type_order'])


def classify_outlier_types(df_outliers: pd.DataFrame) -> pd.DataFrame:
    """
    Classify outliers into 'high_degree' or 'low_degree' based on Deviation > 0.
    """
    logger.info("Classifying outliers into high/low degree types...")
    df_outliers['outlier_type'] = np.where(
        df_outliers['Deviation'] > 0, 
        'high_degree', 
        'low_degree'
    )
    return df_outliers


def get_all_outliers() -> pd.DataFrame:
    """
    Standard loader calling detect_outliers (5th/95th, strict).
    """
    return _get_all_outliers_internal(detect_outliers)


def get_all_outliers_exact() -> pd.DataFrame:
    """
    New loader calling detect_outliers_exact (20th/80th, inclusive).
    Use this for intersection analysis to match Notebook 007 results (786 outliers).
    """
    return _get_all_outliers_internal(detect_outliers_exact)


def _get_all_outliers_internal(detection_func) -> pd.DataFrame:
    """Internal helper to load data and run a specific detection function."""
    logger.info("Loading network and prevalence data...")
    all_data = []
    
    for sex in SEXES:
        for age_id, age_range in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex} data"):
            df = NetworkAnalyzer.load_node_metrics(sex, age_id)
            if not df.empty:
                df = df[(df['Degree'] > 0) & (df['Prevalence'] > 0)].copy()
                all_data.append(df)
    
    if not all_data:
        logger.error("No data loaded.")
        return pd.DataFrame()

    df_all = pd.concat(all_data, ignore_index=True)
    
    # Run the specific detection function passed in
    df_processed = detection_func(df_all)
    
    df_processed = NetworkAnalyzer.add_english_descriptions(df_processed)
    df_processed = classify_outlier_types(df_processed)
    
    return df_processed


# Alias for compatibility
detect_outliers_exact = detect_outliers_exact # (Self-referential, just ensuring name exists)


@app.command()
def main():
    """Main execution function to generate outlier table."""
    logger.info("Starting Outlier Detection Pipeline")
    
    # 1. Standard Analysis (5th/95th)
    df_standard = get_all_outliers()
    if not df_standard.empty:
        logger.info("Generating standard outliers table (5th/95th)...")
        table_data = select_top_outliers(df_standard, n_high=20, n_low=10)
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_file = PROCESSED_DATA_DIR / 'outliers_data_FINAL.csv'
        
        cols = [
            'Sex', 'Age_Group', 'Age_Range', 'ICD_Code', 
            'Description_GER', 'Description_Eng', 
            'Degree', 'Prevalence', 'Ratio', 'Log_ratio', 
            'Deviation', 'Outlier', 'outlier_type', 'Betweenness', 'Mortality'
        ]
        cols = [c for c in cols if c in table_data.columns]
        
        table_data[cols].to_csv(csv_file, index=False)
        logger.success(f"✓ Standard Data CSV saved to: {csv_file}")

    # 2. Exact Analysis (20th/80th, Inclusive)
    logger.info("Generating EXACT outliers for intersection analysis (20th/80th, Inclusive)...")
    df_exact = get_all_outliers_exact()
    
    exact_file = PROCESSED_DATA_DIR / 'Outliers_EXACT.csv'
    
    # Only save actual outliers
    df_exact_outliers = df_exact[df_exact['Outlier'] == True]
    df_exact_outliers.to_csv(exact_file, index=False)
    
    # Verify count for high degree outliers (Positive Deviation)
    n_high_degree = len(df_exact_outliers[df_exact_outliers['Deviation'] > 0])
    
    logger.success(f"✓ Exact Outliers CSV saved to: {exact_file}")
    logger.info(f"Total High Degree Outliers Found: {n_high_degree} (Expected ~786)")

if __name__ == "__main__":
    app()