"""
Bridge visualization module.

This module integrates the plotting logic from '010_panel_low_high_bridges.ipynb',
generating panel plots and heatmaps for high-mortality bridge edges.

It exploits the existing codebase structure for paths, configuration, and logging.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import typer
from pathlib import Path
from loguru import logger
import seaborn as sns

from tapas.config import (
    PROCESSED_DATA_DIR, 
    FIGURES_DIR, 
    SEXES, 
    AGE_GROUPS
)
from tapas.analysis_config import (
    OUTPUT_FILENAMES, 
    DEFAULT_DPI,
    DEFAULT_FIGURE_SIZE
)

app = typer.Typer()

def get_chapter(icd_code: str) -> str:
    """Extracts the ICD-10 Chapter (first letter) from a code."""
    if pd.isna(icd_code) or len(str(icd_code)) == 0:
        return "Unknown"
    return str(icd_code)[0].upper()

def get_chapter_pair(row) -> str:
    """Creates a sorted string representation of the chapter pair."""
    c1 = get_chapter(row['ICD_Code_1'])
    c2 = get_chapter(row['ICD_Code_2'])
    return "-".join(sorted([c1, c2]))

@app.command()
def main():
    """
    Generate visualizations for Bridge Edges (High/Low Mortality).
    Replicates logic from 010_panel_low_high_bridges.ipynb.
    """
    logger.info("Starting Bridge Edge Visualization...")

    # 1. Load Data
    input_file = OUTPUT_FILENAMES.get("bridge_edges", "bridge_edges_mortality_ZSCORE.csv")
    input_path = PROCESSED_DATA_DIR / input_file

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run robustness_analysis.py or ensuring the bridge edges file exists.")
        raise typer.Exit(code=1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} bridge edges.")

    # 2. Process Data: Add Chapter Pairs
    logger.info("Processing chapter pairs...")
    df['Chapter_Pair'] = df.apply(get_chapter_pair, axis=1)

    # Filter to top chapter pairs to avoid color chaos (optional, but good for visualization)
    # Keeping logic simple as requested: assume we plot all or the most frequent ones.
    # To match notebook logic of "legend handles", we assign colors to unique pairs.
    unique_pairs = sorted(df['Chapter_Pair'].unique())
    
    # Create a consistent color palette
    # Using tab20 or similar for distinct categorical colors
    palette = plt.cm.tab20(np.linspace(0, 1, len(unique_pairs)))
    pair_colors = dict(zip(unique_pairs, palette))

    # 3. Create Panel Plot (Scatter/Distribution by Age/Sex)
    # We create a figure with subplots for each Sex
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Ensure OUTPUT directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for idx, sex in enumerate(SEXES):
        ax = axes[idx]
        subset = df[df['Sex'] == sex].copy()
        
        if subset.empty:
            logger.warning(f"No data for {sex}")
            continue

        # Map age groups to numeric indices for plotting
        # AGE_GROUPS keys are integers 1-8.
        # We add jitter to x-axis to visualize overlapping points
        jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
        
        # Plot each point
        # X: Age Group, Y: Mortality Difference (or Z-score)
        # Using Mortality_Diff if available, else Z_Score_Product
        y_col = 'Mortality_Diff' if 'Mortality_Diff' in df.columns else 'Z_Score_Product'
        
        # We iterate pairs to plot them with correct colors (vectorized would be faster but this is clearer for legend logic)
        for pair in unique_pairs:
            pair_data = subset[subset['Chapter_Pair'] == pair]
            if pair_data.empty:
                continue
                
            x_vals = pair_data['Age_Group'] + jitter[pair_data.index.isin(subset.index) & (subset['Chapter_Pair'] == pair)]
            y_vals = pair_data[y_col]
            
            ax.scatter(x_vals, y_vals, 
                      color=pair_colors[pair], 
                      alpha=0.7, 
                      s=60, 
                      edgecolor='white', 
                      linewidth=0.5,
                      label=pair)

        # Formatting
        ax.set_title(f"{sex}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Age Group", fontsize=12)
        ax.set_xticks(list(AGE_GROUPS.keys()))
        ax.set_xticklabels([AGE_GROUPS[k] for k in sorted(AGE_GROUPS.keys())], rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel("Mortality Difference", fontsize=12)

    plt.suptitle("High Mortality Bridge Edges by Chapter Pair", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save Main Panel
    output_panel = FIGURES_DIR / "panel_bridge_edges_scatter.png"
    plt.savefig(output_panel, dpi=DEFAULT_DPI)
    plt.close()
    logger.success(f"Saved Panel Plot to {output_panel}")

    # 4. Create Legend-Only Figure (Logic from Notebook Snippet)
    logger.info("Generating separate legend figure...")
    
    # Create a dedicated figure just for the legend
    fig_leg, ax_leg = plt.subplots(figsize=(6, max(4, len(unique_pairs) * 0.3)))
    ax_leg.axis('off')

    # Create handles manually as per snippet logic
    legend_handles = [
        plt.Line2D([0], [0], color=pair_colors[p], lw=6, label=p) 
        for p in unique_pairs
    ]

    # Split legend into columns if too long
    if len(unique_pairs) > 10:
        mid = (len(legend_handles) + 1) // 2
        h1, l1 = legend_handles[:mid], unique_pairs[:mid]
        h2, l2 = legend_handles[mid:], unique_pairs[mid:]
        
        leg1 = ax_leg.legend(h1, l1, title='Chapter Pair', frameon=False,
                           loc='center left', bbox_to_anchor=(0.0, 0.5))
        ax_leg.add_artist(leg1)
        ax_leg.legend(h2, l2, title='Chapter Pair', frameon=False,
                    loc='center left', bbox_to_anchor=(0.5, 0.5))
    else:
        ax_leg.legend(handles=legend_handles, title='Chapter Pair', frameon=False, loc='center')

    fig_leg.tight_layout()
    
    output_legend = FIGURES_DIR / "legend_bridge_edges_chapter_pair.png"
    plt.savefig(output_legend, dpi=DEFAULT_DPI)
    plt.close()
    logger.success(f"Saved Legend to {output_legend}")

    # 5. Optional: Chapter Pair Heatmap (Counts)
    # A heatmap showing which pairs are most common across ages
    logger.info("Generating Chapter Pair Heatmap...")
    
    # Aggregate data: Count of bridges per (Age, Chapter Pair)
    # Combining sexes for a unified view, or could split
    heatmap_data = df.groupby(['Chapter_Pair', 'Age_Group']).size().unstack(fill_value=0)
    
    # Filter for readability: Top 15 pairs
    top_pairs = heatmap_data.sum(axis=1).sort_values(ascending=False).head(20).index
    heatmap_data = heatmap_data.loc[top_pairs]

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt='d', linewidths=.5)
    plt.title("Frequency of High-Mortality Bridges by Chapter Pair and Age", fontsize=14)
    plt.xlabel("Age Group")
    plt.ylabel("Chapter Pair (ICD-10)")
    
    # Fix x-labels
    current_locs = plt.xticks()[0]
    # Ensure labels match the column names (Age Groups)
    plt.xticks(current_locs, [AGE_GROUPS.get(c, c) for c in heatmap_data.columns], rotation=45)

    plt.tight_layout()
    output_heatmap = FIGURES_DIR / "heatmap_bridge_edges_counts.png"
    plt.savefig(output_heatmap, dpi=DEFAULT_DPI)
    plt.close()
    logger.success(f"Saved Heatmap to {output_heatmap}")

if __name__ == "__main__":
    app()