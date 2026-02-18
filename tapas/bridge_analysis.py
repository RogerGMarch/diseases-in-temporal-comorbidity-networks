"""
Analysis and Visualization of Low-to-High Mortality Bridges.

This script reproduces the plots for 'Low-to-High Mortality Bridges' 
(Figure 3B in the TAPAS paper) using the project's configuration and structure.

It handles:
1. Loading bridge edge data with robust error checking.
2. Plotting the count of bridges by Age and Sex.
3. Plotting the distribution of ICD-10 Chapters involved in bridges.
4. Generating the 'Bridge Heatmap' visualizing specific disease pairs over time.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

# Import configuration from your config file
from tapas.config import PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()

# --- Constants & Configuration ---

# ICD-10 Chapter Colors (Approximate based on paper)
CHAPTER_COLORS = {
    "C": "#8c1515",  # Neoplasms (Red/Brown)
    "E": "#f7b900",  # Endocrine (Yellow)
    "F": "#4b0082",  # Mental (Indigo/Purple)
    "I": "#e95420",  # Circulatory (Orange)
    "J": "#0055a6",  # Respiratory (Blue)
    "K": "#e3001b",  # Digestive (Red)
    "N": "#0099cc",  # Genitourinary (Light Blue)
    "Other": "#808080" # Grey
}

AGE_GROUPS = ["40-49", "50-59", "60-69", "70-79"]


def get_icd_chapter(code: str) -> str:
    """Extract the ICD-10 chapter (first letter) from a code."""
    if not isinstance(code, str) or len(code) == 0:
        return "Other"
    return code[0].upper()


def get_chapter_pair_color(source_code: str, target_code: str) -> str:
    """Determine color based on the target (high mortality) chapter, matching paper style."""
    target_chap = get_icd_chapter(target_code)
    return CHAPTER_COLORS.get(target_chap, CHAPTER_COLORS["Other"])


# --- Data Loading ---

def load_data(filename: str = "bridge_edges_mortality_ZSCORE.csv") -> pd.DataFrame:
    """
    Load the bridge edge dataset with robust column checking.
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    if not file_path.exists():
        logger.error(f"Data file not found at: {file_path}")
        logger.info("Please run 'generate_bridge_stats.py' to generate the required data.")
        return pd.DataFrame()
        
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"File {file_path} is empty.")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"File {file_path} contains no data rows.")
        return df

    # --- Column Mapping / Renaming ---
    # Map external column names to internal standard
    rename_map = {
        "Sex": "sex",
        "Age_Range": "age_group",
        "ICD_Code_1": "source",
        "ICD_Code_2": "target",
        "Edge_Betweenness": "edge_betweenness",
        "Mortality_Diff": "mortality_diff",
        "z_product": "z_score_product",
        "z_mort_diff": "z_mortality_diff"
    }
    df = df.rename(columns=rename_map)

    # --- Robust Column Checking ---
    required_cols = ["source", "target"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        logger.error(f"KeyError Prevention: Missing columns {missing_cols} in {file_path}.")
        logger.error(f"Available columns: {list(df.columns)}")
        logger.info("Ensure 'generate_bridge_stats.py' was run correctly and saved these columns.")
        return pd.DataFrame() # Return empty to skip processing

    # Ensure necessary columns exist (feature engineering if raw)
    # Safe to access 'source' and 'target' now
    if "source_chapter" not in df.columns:
        df["source_chapter"] = df["source"].astype(str).apply(get_icd_chapter)
    if "target_chapter" not in df.columns:
        df["target_chapter"] = df["target"].astype(str).apply(get_icd_chapter)
    if "pair_label" not in df.columns:
        df["pair_label"] = df["source"].astype(str) + "-" + df["target"].astype(str)
    if "chapter_pair" not in df.columns:
        df["chapter_pair"] = df["source_chapter"] + "-" + df["target_chapter"]
        
    return df


# --- Plotting Functions ---

def plot_bridge_counts(df: pd.DataFrame, output_dir: Path):
    """
    Figure 3B (Top Left): Number of bridges by age and sex.
    """
    if df.empty:
        logger.warning("Skipping plot_bridge_counts: No data.")
        return

    # Aggregate data
    counts = df.groupby(["age_group", "sex"]).size().reset_index(name="n_bridges")
    
    # Ensure all ages/sexes are present even if count is 0
    full_index = pd.MultiIndex.from_product([AGE_GROUPS, ["Male", "Female"]], names=["age_group", "sex"])
    counts = counts.set_index(["age_group", "sex"]).reindex(full_index, fill_value=0).reset_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=counts, x="age_group", y="n_bridges", hue="sex", palette={"Female": "#FF69B4", "Male": "#1E90FF"})
    
    plt.title("Number of Low-to-High Mortality Bridges")
    plt.xlabel("Age Group")
    plt.ylabel("Count of Bridges")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Sex")
    
    output_path = output_dir / "bridges_counts_by_age.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved Count Plot: {output_path}")
    plt.close()


def plot_chapter_trajectories(df: pd.DataFrame, output_dir: Path):
    """
    Figure 3B (Bottom Left): Smoothed trajectories of number of bridges by Target Chapter.
    """
    if df.empty:
        logger.warning("Skipping plot_chapter_trajectories: No data.")
        return

    # Count bridges per target chapter per age
    chap_counts = df.groupby(["age_group", "target_chapter"]).size().reset_index(name="count")
    
    plt.figure(figsize=(8, 5))
    
    # Plot a line for each chapter found
    chapters = chap_counts["target_chapter"].unique()
    
    for chap in chapters:
        color = CHAPTER_COLORS.get(chap, CHAPTER_COLORS["Other"])
        subset = chap_counts[chap_counts["target_chapter"] == chap]
        
        # Align with standard age groups x-axis
        # We map age groups to numeric indices for plotting lines
        x_map = {age: i for i, age in enumerate(AGE_GROUPS)}
        subset = subset[subset["age_group"].isin(AGE_GROUPS)].copy()
        subset["x"] = subset["age_group"].map(x_map)
        subset = subset.sort_values("x")
        
        if not subset.empty:
            plt.plot(subset["x"], subset["count"], marker='o', label=f"Chapter {chap}", 
                     color=color, linewidth=2.5, alpha=0.8)

    plt.xticks(range(len(AGE_GROUPS)), AGE_GROUPS)
    plt.title("Evolution of High-Mortality Bridge Targets by Chapter")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Bridges")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = output_dir / "bridges_chapter_trajectories.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved Trajectory Plot: {output_path}")
    plt.close()


def plot_bridge_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Figure 3B (Right): Heatmap/Matrix of Bridge Pairs over Age.
    """
    if df.empty:
        logger.warning("Skipping plot_bridge_heatmap: No data.")
        return

    # Filter for top frequent pairs to avoid clutter (mimicking paper's focused list)
    pair_counts = df["pair_label"].value_counts()
    top_pairs = pair_counts.head(40).index.tolist() # Show top 40 pairs
    
    subset = df[df["pair_label"].isin(top_pairs)].copy()
    
    # Create a pivot table for the heatmap structure: Index=Pair, Columns=Age
    pivot = subset.pivot_table(
        index="pair_label", 
        columns="age_group", 
        values="target_chapter", 
        aggfunc='first' 
    )
    
    # Sort index
    subset["sort_key"] = subset["target_chapter"] + subset["source_chapter"]
    sort_map = subset.set_index("pair_label")["sort_key"].to_dict()
    sorted_pairs = sorted(pivot.index, key=lambda x: sort_map.get(x, ""))
    pivot = pivot.reindex(sorted_pairs)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 12))
    
    y_labels = pivot.index
    x_labels = AGE_GROUPS
    x_map = {label: i for i, label in enumerate(x_labels)}
    
    for y_i, pair in enumerate(y_labels):
        for x_label in x_labels:
            if x_label in pivot.columns:
                chapter = pivot.loc[pair, x_label]
                if pd.notna(chapter):
                    color = CHAPTER_COLORS.get(chapter, CHAPTER_COLORS["Other"])
                    rect = mpatches.Rectangle(
                        (x_map[x_label] - 0.4, y_i - 0.4), 
                        0.8, 0.8, 
                        facecolor=color, 
                        edgecolor='none'
                    )
                    ax.add_patch(rect)

    # Styling
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_title("Critical Low-to-High Mortality Bridges")
    ax.set_xlabel("Age Group")
    
    legend_patches = [mpatches.Patch(color=c, label=f"Chapter {k}") 
                      for k, c in CHAPTER_COLORS.items() if k != "Other"]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Target Chapter")
    
    ax.invert_yaxis() 
    
    output_path = output_dir / "bridges_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved Heatmap Plot: {output_path}")
    plt.close()


@app.command()
def main():
    """
    Run the analysis and generate all bridge plots.
    """
    # 1. Setup
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Data
    df = load_data()
    
    if df.empty:
        logger.warning("Dataframe is empty. Plots will not be generated.")
        return

    logger.info(f"Loaded {len(df)} bridge edges.")

    # 3. Generate Plots
    logger.info("Generating Bridge Counts Plot...")
    plot_bridge_counts(df, FIGURES_DIR)
    
    logger.info("Generating Chapter Trajectories Plot...")
    plot_chapter_trajectories(df, FIGURES_DIR)
    
    logger.info("Generating Bridge Pair Heatmap...")
    plot_bridge_heatmap(df, FIGURES_DIR)
    
    logger.success("All plots generated successfully.")


if __name__ == "__main__":
    app()