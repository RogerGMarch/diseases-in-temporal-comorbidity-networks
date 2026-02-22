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

# ── Style — matches panel_outliers.py / Figure 2 ──────────────────────────────
plt.rcParams["font.family"] = "Ubuntu"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
    }
)

# ICD chapter colour palette (same mapping as _CHAPTER_COLOR_MAP in panel_outliers.py)
_HEX_COLORS = [
    "#1AF239", "#58F21A", "#961D1A", "#B41AF2", "#FFC801", "#581AF2",
    "#1AF295", "#1A95F2", "#F2761A", "#1A39F2", "#F21A1A", "#F21AD3",
    "#B4F21A", "#1AF2F2",
]
_CHAPTER_COLOR_MAP = dict(zip(list("ABCDEFGHIJKLMN"), _HEX_COLORS))

# E78 → chapter E, F17 → chapter F
TARGETS = {
    "E78": _CHAPTER_COLOR_MAP["E"],
    "F17": _CHAPTER_COLOR_MAP["F"],
}

SEX_LINESTYLE = {"Female": "-",  "Male": "--"}

# GEXF files carry the correct ICD-code → matrix-row mapping
_GEXF_DIR = INTERIM_DATA_DIR / "extracted" / "Data" / "4.Graphs-gexffiles"
_ADJ_DIR  = INTERIM_DATA_DIR / "extracted" / "Data" / "3.AdjacencyMatrices"
_GEXF_NS  = {"g": "http://www.gexf.net/1.3"}


def _save_figure(fig: plt.Figure, name_base: str) -> None:
    """Save figure to PNG and PDF — same settings as save_figure() in panel_outliers.py."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"{name_base}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight", transparent=True)
    logger.info(f"Saved: {FIGURES_DIR / name_base}.png")


def _gexf_icd_to_row(sex: str, age_id: int) -> dict[str, int]:
    """Return {icd_code: matrix_row_index} using the GEXF node id attribute (1-based → 0-based)."""
    path = _GEXF_DIR / f"Graph_{sex}_ICD_Age_{age_id}.gexf"
    if not path.exists():
        raise FileNotFoundError(path)
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.findall(".//g:node", _GEXF_NS)
    return {n.get("label"): int(n.get("id")) - 1 for n in nodes}


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
    Two-panel figure (Degree | Clustering Coefficient) styled to match Figure 2.
    """
    age_order = [AGE_GROUPS[k] for k in sorted(AGE_GROUPS.keys())]
    age_sort  = {v: k for k, v in AGE_GROUPS.items()}
    metrics   = ["Degree", "ClusteringCoefficient"]
    titles    = {"Degree": "Degree", "ClusteringCoefficient": "Clustering Coefficient"}
    ylabels   = {"Degree": "Degree", "ClusteringCoefficient": "Clustering Coefficient"}

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.0, 3.0),
        gridspec_kw={"wspace": 0.38},
    )

    x_pos = list(range(len(age_order)))

    for ax, metric in zip(axes, metrics):
        for icd, color in TARGETS.items():
            for sex in SEXES:
                sub = (
                    df[(df["ICD_Code"] == icd) & (df["Sex"] == sex)]
                    .copy()
                    .assign(_sort=lambda d: d["Age_Range"].map(age_sort))
                    .sort_values("_sort")
                )
                if sub.empty:
                    continue

                ax.plot(
                    x_pos[: len(sub)],
                    sub[metric].values,
                    color=color,
                    linestyle=SEX_LINESTYLE[sex],
                    linewidth=1.2,
                    marker="o",
                    markersize=4,
                    markeredgewidth=0.3,
                    markeredgecolor="#333333" if sex == "Female" else "none",
                    alpha=1.0 if sex == "Female" else 0.75,
                )

        ax.set_title(titles[metric], fontsize=8, pad=4, loc="left", fontweight="bold")
        ax.set_xlabel("Age Group", fontsize=7)
        ax.set_ylabel(ylabels[metric], fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(age_order, fontsize=7, rotation=45, ha="right")
        ax.tick_params(labelsize=7, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_bounds(x_pos[0], x_pos[-1])

    # ── Legend ────────────────────────────────────────────────────────────────
    icd_handles = [
        mlines.Line2D([], [], color=color, marker="o", markersize=4,
                      markeredgewidth=0.3, markeredgecolor="#333333",
                      linewidth=1.2, label=icd)
        for icd, color in TARGETS.items()
    ]
    sex_handles = [
        mlines.Line2D([], [], color="#555555", linestyle=ls,
                      linewidth=1.2, label=sex)
        for sex, ls in SEX_LINESTYLE.items()
    ]

    leg_icd = axes[1].legend(
        handles=icd_handles,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        handletextpad=0.4,
    )
    axes[1].add_artist(leg_icd)
    axes[1].legend(
        handles=sex_handles,
        title="Sex",
        title_fontsize=7,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.55),
        handletextpad=0.4,
    )

    _save_figure(fig, output_name)
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
