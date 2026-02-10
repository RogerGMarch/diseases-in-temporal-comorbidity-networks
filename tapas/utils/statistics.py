"""
Statistical utility functions for network analysis.

This module consolidates all statistical calculations used across the project,
providing a single source of truth for z-score calculations, log-ratios, and
other statistical transformations.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_z_score(values: pd.Series) -> pd.Series:
    """
    Compute standard Z-scores for a series of values.

    Z-score = (x - mean) / std

    Args:
        values: Series of numeric values

    Returns:
        Series of Z-scores. If std is 0, returns series of zeros.

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> z_scores = compute_z_score(data)
    """
    mean = values.mean()
    std = values.std()
    if std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def modified_zscore(x: float, median: float, mad: float) -> Optional[float]:
    """
    Compute modified z-score using Median Absolute Deviation (MAD).

    Modified Z-score = 0.6745 * (x - median) / MAD

    The constant 0.6745 is the 0.75th quartile of the standard normal distribution,
    making the MAD-based z-score comparable to the standard z-score for normal data.

    This is more robust to outliers than the standard z-score.

    Args:
        x: Value to compute z-score for
        median: Median of the distribution
        mad: Median Absolute Deviation

    Returns:
        Modified z-score, or None if MAD is 0

    References:
        Iglewicz, B., & Hoaglin, D. C. (1993). How to detect and handle outliers.
        ASQC Quality Press.

    Examples:
        >>> median = 5.0
        >>> mad = 1.4826
        >>> modified_zscore(7.0, median, mad)
        0.9096
    """
    if mad == 0:
        return np.nan
    return 0.6745 * (x - median) / mad


def compute_log_ratio(degree: float, prevalence: float, base: int = 10) -> Optional[float]:
    """
    Compute log-ratio: log(degree / prevalence).

    This metric quantifies how connected a disease is relative to its prevalence.
    - Positive values: disease is more connected than expected from prevalence
    - Negative values: disease is less connected than expected from prevalence

    Args:
        degree: Node degree (number of connections)
        prevalence: Disease prevalence (proportion of population)
        base: Logarithm base (default: 10 for log10)

    Returns:
        Log-ratio value, or None if degree <= 0 or prevalence <= 0

    Examples:
        >>> compute_log_ratio(100, 0.01)  # High degree, low prevalence
        5.0  # log10(100/0.01) = log10(10000) = 4.0
    """
    if degree > 0 and prevalence > 0:
        if base == 10:
            return np.log10(degree / prevalence)
        elif base == np.e:
            return np.log(degree / prevalence)
        else:
            return np.log(degree / prevalence) / np.log(base)
    return None


def compute_z_product(z_score1: pd.Series, z_score2: pd.Series) -> pd.Series:
    """
    Compute product of two z-score series.

    This is used for identifying nodes/edges that are simultaneously high
    in two dimensions (e.g., high betweenness AND high mortality).

    The product acts as a logical AND - only positive when both z-scores are positive.

    Args:
        z_score1: First z-score series
        z_score2: Second z-score series

    Returns:
        Product of z-scores

    Examples:
        >>> z_bet = pd.Series([1.5, 2.0, -0.5])
        >>> z_mort = pd.Series([2.0, 1.0, 1.5])
        >>> compute_z_product(z_bet, z_mort)
        [3.0, 2.0, -0.75]
    """
    return z_score1 * z_score2


def compute_geometric_mean_zscore(z_score1: pd.Series, z_score2: pd.Series) -> pd.Series:
    """
    Compute geometric mean of two positive z-scores.

    Geometric mean = sqrt(z1 * z2) when both are positive, 0 otherwise

    This is used for ranking purposes when both dimensions must be above average.

    Args:
        z_score1: First z-score series
        z_score2: Second z-score series

    Returns:
        Geometric mean of z-scores (0 when either is non-positive)

    Examples:
        >>> z_bet = pd.Series([1.5, 2.0, -0.5])
        >>> z_mort = pd.Series([2.0, 1.0, 1.5])
        >>> compute_geometric_mean_zscore(z_bet, z_mort)
        [1.732, 1.414, 0.0]
    """
    return np.where(
        (z_score1 > 0) & (z_score2 > 0),
        np.sqrt(z_score1 * z_score2),
        0
    )


def identify_percentile_outliers(
    values: pd.Series,
    lower_percentile: float = 0.20,
    upper_percentile: float = 0.80
) -> pd.DataFrame:
    """
    Identify outliers using percentile thresholds.

    Args:
        values: Series of values to check for outliers
        lower_percentile: Lower percentile threshold (default: 0.20 for 20th percentile)
        upper_percentile: Upper percentile threshold (default: 0.80 for 80th percentile)

    Returns:
        DataFrame with columns:
        - value: original values
        - is_low_outlier: True if below lower percentile
        - is_high_outlier: True if above upper percentile
        - is_outlier: True if either low or high outlier

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> outliers = identify_percentile_outliers(data, 0.2, 0.8)
    """
    lower_bound = values.quantile(lower_percentile)
    upper_bound = values.quantile(upper_percentile)

    return pd.DataFrame({
        'value': values,
        'is_low_outlier': values < lower_bound,
        'is_high_outlier': values > upper_bound,
        'is_outlier': (values < lower_bound) | (values > upper_bound),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
    })
