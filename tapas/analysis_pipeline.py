import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

# --- Configuration matching Paper ---
AGE_GROUPS = {
    1: "0-9", 2: "10-19", 3: "20-29", 4: "30-39",
    5: "40-49", 6: "50-59", 7: "60-69", 8: "70-79"
}
SEXES = ["Female", "Male"]

# --- Core Statistical Formulas from Paper ---

def compute_log_ratio(degree: pd.Series, prevalence: pd.Series) -> pd.Series:
    """
    Paper Formula: Log-Ratio = log10(degree / prevalence)
    """
    # Avoid division by zero
    ratio = degree.div(prevalence.replace(0, np.nan))
    return np.log10(ratio)

def compute_z_score(series: pd.Series) -> pd.Series:
    """
    Paper Formula: z = (x - mean) / std
    """
    if series.std() == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / series.std()

# --- Method 1: Outlier Detection ---

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements: "identified nodes in the outer quintiles 
    (below the 20th percentile and above the 80th percentile)"
    """
    df = df.copy()
    df['log_ratio'] = compute_log_ratio(df['degree'], df['prevalence'])
    
    # Paper implies calculating quintiles within each stratum
    # Assuming df passed here is already stratified (one age/sex group)
    
    lower_q = df['log_ratio'].quantile(0.20)
    upper_q = df['log_ratio'].quantile(0.80)
    
    df['is_outlier_low'] = df['log_ratio'] <= lower_q
    df['is_outlier_high'] = df['log_ratio'] >= upper_q
    
    return df

# --- Method 2: High-Mortality Sinks ---

def identify_sinks(df: pd.DataFrame, top_percent: float = 0.20) -> pd.DataFrame:
    """
    Implements: "product of the Z-Scores... selected the top 20%"
    Metric: Betweenness * Mortality
    """
    df = df.copy()
    df['z_betweenness'] = compute_z_score(df['betweenness'])
    df['z_mortality'] = compute_z_score(df['mortality'])
    
    df['z_product'] = df['z_betweenness'] * df['z_mortality']
    
    # Implicit logic: We want HIGH centrality and HIGH mortality.
    # We filter for positive Z-scores to ensure we don't select 
    # low-centrality/low-mortality nodes (which would have positive products).
    candidates = df[(df['z_betweenness'] > 0) & (df['z_mortality'] > 0)]
    
    if candidates.empty:
        return pd.DataFrame()
        
    threshold = candidates['z_product'].quantile(1.0 - top_percent)
    return candidates[candidates['z_product'] >= threshold]

# --- Method 3: High-Mortality Bridges ---

def identify_bridges(df_edges: pd.DataFrame, top_percent: float = 0.05, min_diff: float = 0.30) -> pd.DataFrame:
    """
    Implements: "absolute mortality difference exceeded 30%... selected edges in the top 5%"
    Metric: Edge Betweenness * Mortality Difference
    """
    df = df_edges.copy()
    
    # Constraint 1: Mortality difference >= 30%
    df['mortality_diff'] = (df['mortality_u'] - df['mortality_v']).abs()
    candidates = df[df['mortality_diff'] >= min_diff].copy()
    
    if candidates.empty:
        return pd.DataFrame()
        
    # Constraint 2: Z-Score Product
    candidates['z_edge_bet'] = compute_z_score(candidates['edge_betweenness'])
    candidates['z_mort_diff'] = compute_z_score(candidates['mortality_diff'])
    candidates['z_product'] = candidates['z_edge_bet'] * candidates['z_mort_diff']
    
    # Filter for positive deviations (High centrality AND High difference)
    candidates = candidates[(candidates['z_edge_bet'] > 0) & (candidates['z_mort_diff'] > 0)]
    
    if candidates.empty:
        return pd.DataFrame()

    # Selection: Top 5%
    threshold = candidates['z_product'].quantile(1.0 - top_percent)
    return candidates[candidates['z_product'] >= threshold]