"""
Analysis subpackage for comorbidity network analyses.

This package contains specialized analysis modules:
- outliers: Degree-prevalence outlier detection
- mortality: High-mortality sinks identification
- bridges: Critical bridge edges analysis
- critical_nodes: Intersection of high-degree and high-mortality nodes
"""

from tapas.analysis.outliers import (
    detect_outliers_exact,
    select_top_outliers,
    identify_high_mortality_sinks_zscore,
)

__all__ = [
    "detect_outliers_exact",
    "select_top_outliers",
    "identify_high_mortality_sinks_zscore",
]
