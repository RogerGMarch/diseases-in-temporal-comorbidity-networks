#!/usr/bin/env python
"""
Generate Figure 2 (Female) and Fig. S2 (Male) outlier scatter panels.

Usage (with uv):
    uv run python scripts/figure2_outlier_scatter.py              # Both sexes
    uv run python scripts/figure2_outlier_scatter.py --sex Female  # Female only
    uv run python scripts/figure2_outlier_scatter.py --sex Male    # Male only
    uv run python scripts/figure2_outlier_scatter.py --n-labels 8  # More labels
"""

from typing import Optional

import typer

from tapas.analysis.panel_outliers import (
    generate_all_outlier_scatter_panels,
    generate_outlier_scatter_panel,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    sex: Optional[str] = typer.Option(
        None,
        help="Sex to plot: 'Female' (Figure 2) or 'Male' (Fig. S2). "
        "Omit to generate both.",
    ),
    n_labels: int = typer.Option(
        6,
        help="Number of ICD code labels to show per panel.",
    ),
) -> None:
    """Generate the outlier scatter panel figures."""
    if sex is None:
        generate_all_outlier_scatter_panels()
    else:
        generate_outlier_scatter_panel(sex=sex, n_labels=n_labels)


if __name__ == "__main__":
    app()
