"""
Visualization entry point.

Aggregates all plot generation functions into a single ``generate_all_plots``
call used by the main pipeline.
"""

from loguru import logger

from tapas.analysis.panel_outliers import main as generate_panel_plots


def generate_all_plots() -> None:
    """Generate every figure produced by the project.

    Currently includes:
    - Panel bar/alluvial/heatmap plots (sinks, outliers, bridges)
    - Outlier scatter panels (Figure 2 + Fig. S2)

    Individual generators are wrapped in try/except so a single failure
    does not block the remaining plots.
    """
    logger.info("Generating all project plots...")

    # Panel plots (bar charts, alluvial, heatmaps, and scatter panels)
    try:
        generate_panel_plots()
    except Exception as e:
        logger.warning(f"Panel plots failed: {e}")

    logger.success("All plot generation complete.")
