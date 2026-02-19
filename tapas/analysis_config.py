"""
Central configuration for all analysis parameters.

This module defines default parameters used across various analyses to ensure
consistency and reproducibility. All threshold values, percentiles, and other
analysis parameters should be defined here rather than hardcoded in scripts.

Modifying these values will affect the analysis results. Always document
why parameters were chosen and cite the paper section if applicable.
"""

# ============================================================================
# Outlier Detection Parameters
# ============================================================================

# Percentile thresholds for degree-prevalence outlier detection
# Paper Reference: Supplementary Material, Table S1
OUTLIER_LOWER_PERCENTILE = 0.20  # 20th percentile - low degree outliers
OUTLIER_UPPER_PERCENTILE = 0.80  # 80th percentile - high degree outliers

# Number of top outliers to report per sex-age group
OUTLIER_TOP_HIGH = 20  # High-degree outliers
OUTLIER_TOP_LOW = 10   # Low-degree outliers


# ============================================================================
# High-Mortality Sinks Parameters
# ============================================================================

# Percentile threshold for identifying high-mortality sinks
# Nodes with both high betweenness AND high mortality
# Paper Reference: Section on "High-mortality sinks"
HIGH_MORTALITY_TOP_PERCENT = 0.20  # Top 20% by z-score product


# ============================================================================
# Critical Bridge Edges Parameters
# ============================================================================

# Percentile threshold for identifying critical bridge edges
# Edges with high betweenness AND high mortality difference
# Paper Reference: Section on "High-mortality bridges"
BRIDGE_TOP_PERCENT = 0.05  # Top 5% by z-score product

# Minimum absolute mortality difference to consider an edge
# Filters out edges where mortality difference is too small to be meaningful
MIN_MORTALITY_DIFF = 0.30  # 30% minimum difference


# ============================================================================
# Critical Nodes Intersection Parameters
# ============================================================================

# Percentile threshold for mortality sinks when finding intersection
# More permissive than standalone analysis to capture more candidates
CRITICAL_NODES_SINK_PERCENT = 0.40  # Top 40% for intersection analysis


# ============================================================================
# Network Analysis Parameters
# ============================================================================

# Whether to filter isolated nodes (nodes with degree 0) in network metrics
# Paper methodology: "We filtered for nodes with at least one neighbor"
FILTER_ISOLATED_NODES = True

# Whether to use largest connected component for path length calculations
# Disconnected graphs have infinite path lengths between components
USE_LARGEST_COMPONENT = True

# Betweenness centrality normalization
# If True, normalizes by number of node pairs: 1/((n-1)(n-2))
# Paper uses normalized betweenness for individual node analysis
NORMALIZE_BETWEENNESS = True

# Edge betweenness centrality normalization
# If True, normalizes by number of node pairs
NORMALIZE_EDGE_BETWEENNESS = True


# ============================================================================
# Data Parameters
# ============================================================================

# Default year for prevalence data
# Paper uses most recent year available in dataset
PREVALENCE_YEAR = 2014

# Minimum prevalence threshold for inclusion
# Diseases with prevalence below this are excluded from analysis
MIN_PREVALENCE = 0.0  # No minimum by default

# Age group definitions (consistent with paper)
# Maps age_group_id to age range string
AGE_GROUP_LABELS = {
    1: "0-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70-79",
}

# Sex labels
SEX_LABELS = ["Female", "Male"]


# ============================================================================
# Statistical Parameters
# ============================================================================

# Constant for modified z-score calculation
# 0.6745 is the 0.75th quartile of standard normal distribution
# Makes MAD-based z-score comparable to standard z-score
MODIFIED_ZSCORE_CONSTANT = 0.6745

# Logarithm base for log-ratio calculations
# Paper uses log base 10 for interpretability
LOG_RATIO_BASE = 10


# ============================================================================
# Robustness Analysis Parameters
# ============================================================================

# Edge weight thresholds for robustness testing
# Tests network properties at different edge weight cutoffs
ROBUSTNESS_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================================
# Visualization Parameters
# ============================================================================

# Default figure size (width, height) in inches
DEFAULT_FIGURE_SIZE = (10, 6)

# Default DPI for saved figures
DEFAULT_DPI = 300

# Default color palette for plots
# Uses colorblind-friendly palette
DEFAULT_PALETTE = "colorblind"


# ============================================================================
# Output Parameters
# ============================================================================

# Default output filenames for various analyses
OUTPUT_FILENAMES = {
    "outliers": "outliers_data_S1.csv",
    "mortality_sinks": "high_mortality_sinks_ZSCORE.csv",
    "bridge_edges": "bridge_edges_mortality_ZSCORE.csv",
    "critical_nodes": "critical_nodes_intersection_ZSCORE.csv",
    "network_properties": "network_properties_table.csv",
    "network_table1": "network_properties_table1_format.csv",
    "prevalence_degree": "prevalence_degree_analysis.csv",
    "high_risk_nodes": "high_risk_nodes.csv",
    "high_risk_edges": "high_risk_edges.csv",
    "table2_critical": "table2_critical_diseases.csv",
}

# Whether to include index column in output CSV files
CSV_INCLUDE_INDEX = False

# Float precision for output CSV files
CSV_FLOAT_FORMAT = "%.6f"
