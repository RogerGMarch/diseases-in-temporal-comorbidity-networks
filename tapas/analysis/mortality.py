"""
High-mortality sinks analysis.

This module identifies diseases (nodes) that are both:
1. Central in the comorbidity network (high betweenness centrality)
2. Associated with high mortality rates

These are imported from the consolidated outliers module.
"""

from tapas.analysis.outliers import identify_high_mortality_sinks_zscore

__all__ = ["identify_high_mortality_sinks_zscore"]
