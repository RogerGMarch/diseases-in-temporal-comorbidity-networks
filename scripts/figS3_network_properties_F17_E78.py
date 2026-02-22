#!/usr/bin/env python
"""
Figure S3: Network properties of nicotine dependence (F17) and
lipoprotein metabolism disorders (E78).

Plots the evolution of Degree and Clustering Coefficient across age groups
for both ICD codes, with Female (solid) and Male (dashed) lines.

Usage:
    uv run python scripts/figS3_network_properties_F17_E78.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tapas.config import (
    AGE_GROUPS,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    SEXES,
)

# GEXF files carry the correct ICD-code → matrix-row mapping
_GEXF_DIR = INTERIM_DATA_DIR / "extracted" / "Data" / "4.Graphs-gexffiles"
_ADJ_DIR  = INTERIM_DATA_DIR / "extracted" / "Data" / "3.AdjacencyMatrices"
_GEXF_NS  = {"g": "http://www.gexf.net/1.3"}


def _gexf_icd_to_row(sex: str, age_id: int) -> dict[str, int]:
    """Return {icd_code: matrix_row_index} using the GEXF node id attribute (1-based → 0-based)."""
    path = _GEXF_DIR / f"Graph_{sex}_ICD_Age_{age_id}.gexf"
    if not path.exists():
        raise FileNotFoundError(path)
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.findall(".//g:node", _GEXF_NS)
    return {n.get("label"): int(n.get("id")) - 1 for n in nodes}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Ubuntu"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
    }
)

# ICD codes of interest and their display colors
TARGETS = {
    "E78": "#E8A020",   # gold/yellow
    "F17": "#5B4EC8",   # violet/purple
}

SEX_LINESTYLE = {"Female": "-", "Male": "--"}
SEX_LABEL     = {"Female": "Female", "Male": "Male"}


# ── Data collection ───────────────────────────────────────────────────────────

def collect_node_metrics_with_clustering() -> pd.DataFrame:
    """
    Load per-node Degree and (local) Clustering Coefficient for E78 / F17
    across every sex × age-group combination.

    The GEXF files store the node labels (ICD codes) in the same row order as
    the adjacency matrix CSVs, so we use the GEXF to build the ICD→row mapping.
    """
    records = []

    for sex in SEXES:
        for age_id, age_label in AGE_GROUPS.items():
            adj_path = _ADJ_DIR / f"Adj_Matrix_{sex}_ICD_age_{age_id}.csv"
            if not adj_path.exists():
                logger.warning(f"Missing: {adj_path}")
                continue

            try:
                icd_to_row = _gexf_icd_to_row(sex, age_id)
            except FileNotFoundError as e:
                logger.warning(f"Missing GEXF: {e}")
                continue

            try:
                A = np.loadtxt(adj_path, delimiter=" ")
                G = nx.from_numpy_array(A)
                # Relabel nodes with ICD codes using the id-based mapping
                row_to_icd = {v: k for k, v in icd_to_row.items()}
                G = nx.relabel_nodes(G, row_to_icd)
            except Exception as e:
                logger.error(f"Could not load graph {sex} {age_label}: {e}")
                continue

            clustering = nx.clustering(G)
            degrees    = dict(G.degree())

            for code in TARGETS:
                if code not in G:
                    continue
                records.append(
                    {
                        "Sex": sex,
                        "Age_Group": age_id,
                        "Age_Range": age_label,
                        "ICD_Code": code,
                        "Degree": degrees[code],
                        "ClusteringCoefficient": clustering[code],
                    }
                )
            logger.info(f"  Loaded {sex} age {age_label}")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            "No data found for E78 / F17. "
            "Check that GEXF and adjacency matrix files exist."
        )
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figS3(df: pd.DataFrame, output_name: str = "figS3_network_properties_F17_E78") -> None:
    """
    Two-panel figure (Degree | Clustering Coefficient) with one line per
    ICD-code × sex combination.
    """
    age_order  = [AGE_GROUPS[k] for k in sorted(AGE_GROUPS.keys())]
    metrics    = ["Degree", "ClusteringCoefficient"]
    titles     = {"Degree": "Degree", "ClusteringCoefficient": "ClusteringCoefficient"}
    ylabels    = {"Degree": "Value", "ClusteringCoefficient": "Value"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, metric in zip(axes, metrics):
        for icd, color in TARGETS.items():
            for sex in SEXES:
                sub = df[(df["ICD_Code"] == icd) & (df["Sex"] == sex)].copy()
                if sub.empty:
                    continue

                # Sort by age group order
                sub["_sort"] = sub["Age_Range"].map({v: k for k, v in AGE_GROUPS.items()})
                sub = sub.sort_values("_sort")

                ax.plot(
                    sub["Age_Range"],
                    sub[metric],
                    color=color,
                    linestyle=SEX_LINESTYLE[sex],
                    linewidth=1.8,
                    marker="o",
                    markersize=5,
                    label=f"{icd} – {SEX_LABEL[sex]}",
                )

        ax.set_title(titles[metric], fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel(ylabels[metric])
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels(age_order, rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="lightgrey")

    # ── Shared legend (right side) ────────────────────────────────────────────
    # ICD code handles
    icd_handles = [
        mlines.Line2D([], [], color=color, marker="o", markersize=5,
                      linewidth=1.8, label=icd)
        for icd, color in TARGETS.items()
    ]
    # Sex handles
    sex_handles = [
        mlines.Line2D([], [], color="grey", linestyle=ls, linewidth=1.8, label=lbl)
        for sex, ls in SEX_LINESTYLE.items()
        for lbl in [SEX_LABEL[sex]]
    ]

    legend_icd = axes[1].legend(
        handles=icd_handles,
        title=None,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    axes[1].add_artist(legend_icd)
    axes[1].legend(
        handles=sex_handles,
        title="Sex",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.60),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"{output_name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.success(f"Saved: {out}")

    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Collecting node metrics for E78 and F17 …")
    df = collect_node_metrics_with_clustering()
    logger.info(f"  {len(df)} records collected.")
    logger.info("Plotting Figure S3 …")
    plot_figS3(df)
    logger.success("Done.")


if __name__ == "__main__":
    main()
