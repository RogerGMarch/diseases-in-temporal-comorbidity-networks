"""
Statistical utility functions for Tapas analysis.
"""
import pandas as pd
import numpy as np

def compute_z_score(series: pd.Series) -> pd.Series:
    """
    Computes standard Z-score: (x - mean) / std.
    Returns 0s if standard deviation is 0.
    """
    if series.empty:
        return series
    std = series.std()
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def modified_zscore(series: pd.Series) -> pd.Series:
    """
    Computes Modified Z-score using Median Absolute Deviation (MAD).
    Robust against outliers.
    Formula: 0.6745 * (x - median) / MAD
    
    This function is designed to be used with pandas transform:
    df.groupby('...')['val'].transform(modified_zscore)
    """
    if series.empty:
        return series
        
    median = series.median()
    diff = (series - median).abs()
    mad = diff.median()
    
    if mad == 0:
        return pd.Series(0.0, index=series.index)
        
    return 0.6745 * (series - median) / mad

def compute_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Computes log10(numerator / denominator) safely.
    Returns NaN where ratio <= 0, denominator == 0, or inputs are invalid.
    """
    # Ensure inputs are numeric
    num = pd.to_numeric(numerator, errors='coerce')
    denom = pd.to_numeric(denominator, errors='coerce')
    
    # Calculate ratio safely, handling division by zero
    # div returns inf if denom is 0, which we'll handle next
    ratio = num.div(denom)
    
    # Replace infs with NaN
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    # Log transform: only valid for strictly positive ratios
    # Mask values <= 0
    ratio[ratio <= 0] = np.nan
    
    return np.log10(ratio)