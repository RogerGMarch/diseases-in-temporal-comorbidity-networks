"""
Panel Visualization: Outliers, Sinks, and Bridges.

Generates panel figures comparing counts by Age and Sex:
1. High-Degree Outliers (Hubs) - Bar chart.
2. High-Mortality Sinks - Bar chart.
3. Bridge Edges - Bar chart.
4. Mortality Sinks Alluvial Flow - Ribbon plot showing ICD Chapter distribution across ages.

All plots share specific visual styles (pastel colors, dimensions, spine adjustments).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
import typer
from loguru import logger
from pathlib import Path
from typing import Optional, List, Dict

from tapas.config import PROCESSED_DATA_DIR, FIGURES_DIR, DATA_DIR

app = typer.Typer()

# Set style parameters globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 8, 
    'axes.titlesize': 8, 
    'axes.labelsize': 8, 
    'xtick.labelsize': 8, 
    'ytick.labelsize': 8, 
    'legend.fontsize': 8
})

# Specific colors
SEX_COLORS = {'Male': '#1FA3FF', 'Female': '#FF5A8A'}

def save_figure(fig, name_base):
    """Save figure to png and pdf with high dpi."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name_base}.png"
    pdf_path = FIGURES_DIR / f"{name_base}.pdf"
    
    fig.savefig(png_path, dpi=600, bbox_inches='tight', transparent=True)
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', transparent=True)
    
    logger.info(f"Saved: {png_path}")

def generate_bar_panel(df: pd.DataFrame, output_name: str, y_label: str):
    """
    Generic function to generate the Age/Sex bar chart with specific styling.
    """
    # 1. Normalize column names
    col_map = {c.lower(): c for c in df.columns}
    sex_col = col_map.get('sex') or col_map.get('gender') or col_map.get('Sex')
    age_col = col_map.get('age_group') or col_map.get('Age_Group')
    
    if not sex_col or not age_col:
        logger.error(f"Could not identify Sex/Age columns in {output_name}. Found: {df.columns}")
        return

    # 2. Prepare Data
    df_plot = df.copy()
    
    # Convert age group like 'age_1' -> 1 if necessary
    if df_plot[age_col].dtype == object:
        df_plot['age_num'] = df_plot[age_col].astype(str).str.replace('age_', '', regex=False).astype(int)
    else:
        df_plot['age_num'] = df_plot[age_col].astype(int)

    # Aggregate Counts
    counts = (
        df_plot.groupby([sex_col, 'age_num'])
        .size()
        .reset_index(name='count')
    )
    
    if counts.empty:
        logger.warning(f"No data to plot for {output_name}")
        return

    # 3. Generate Plot
    fig, ax = plt.subplots(figsize=(2.00, 1.16))
    
    for sex, sub in counts.groupby(sex_col):
        # Determine color and offset
        sex_str = str(sex)
        color = SEX_COLORS.get(sex_str, '#CCCCCC')
        is_male = sex_str.lower().startswith('m')
        offset = 0.15 if is_male else -0.15
        
        ax.bar(
            sub['age_num'] + offset,
            sub['count'], 
            width=0.3, 
            label=sex_str, 
            color=color
        )

    # 4. Styling
    age_labels = ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70-79']
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(age_labels, rotation=45, ha='right', fontsize=8)
    
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Age', fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    
    # Legend settings
    ax.legend(frameon=False, fontsize=8, title_fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.15))

    # Bound axes
    max_count = counts['count'].max()
    ax.set_xlim(0.5, 8.5)
    # Add +2 padding for visual breathing room
    ax.set_ylim(0, max_count + 2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set spine bounds (Data Limits)
    ax.spines['bottom'].set_bounds(1, 8)
    ax.spines['left'].set_bounds(0, max_count + 2)

    save_figure(fig, output_name)
    plt.close(fig) # Close to free memory

# -------------------------------------------------------------------------
# Alluvial / Ribbon Plot Helpers
# -------------------------------------------------------------------------

def ribbon_path(x0, b0, t0, x1, b1, t1, bend=0.55):
    """Helper: curved ribbon (cubic Bezier)"""
    dx = x1 - x0
    cdx = dx * bend

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,  # top edge
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,  # bottom edge
        MplPath.CLOSEPOLY
    ]
    verts = [
        (x0, t0),
        (x0 + cdx, t0),
        (x1 - cdx, t1),
        (x1, t1),
        (x1, b1),
        (x1 - cdx, b1),
        (x0 + cdx, b0),
        (x0, b0),
        (x0, t0)
    ]
    return MplPath(verts, codes)

def generate_alluvial_plot(df_in: pd.DataFrame, output_name: str):
    """
    Generates an alluvial/ribbon plot showing the flow of ICD chapters across age groups.
    """
    logger.info(f"Generating Alluvial Plot: {output_name}...")
    
    # Prepare DataFrame
    df = df_in.copy()
    
    # Ensure necessary columns exist
    # If ICD_Code column names vary, try to normalize
    col_map = {c.lower(): c for c in df.columns}
    icd_col = col_map.get('icd_code')
    age_col = col_map.get('age_group') or col_map.get('age_range') # Using 'Age_Group' (int) or 'Age_Range' (str)
    
    # Fallback for Age_Group logic from main dataframe loading (typically 'Age_Group' is int 1-8)
    if 'Age_Group' in df.columns:
        df['age_num'] = df['Age_Group'].astype(int)
    elif 'age_num' not in df.columns:
        # Try to parse from whatever age column exists
        logger.error("Could not determine 'age_num' for alluvial plot.")
        return

    if icd_col:
        # Extract Chapter (First letter)
        df["ICD_Chapter"] = df[icd_col].astype(str).str[0]
    else:
        logger.error("Could not find ICD Code column for alluvial plot.")
        return

    # -----------------------------
    # Data -> wide matrix
    # -----------------------------
    # Ensure ICD_Chapter is first char of ICD_Code (already done above, but good to verify)
    if "ICD_Chapter" not in df.columns:
         df["ICD_Chapter"] = df[icd_col].astype(str).str[0]

    counts = (
        df.groupby(["age_num", "ICD_Chapter"])
          .size()
          .reset_index(name="count"))
    
    ages = sorted(counts["age_num"].unique())
    chapters = sorted(counts["ICD_Chapter"].unique())

    # Define palette using the specific hex colors provided
    hex_colors = [
      '#1AF239',  # (26, 242, 57) A
      '#58F21A',  # (88, 242, 26) B
      '#961D1A',  # (150, 29, 26) C
      '#B41AF2',  # (180, 26, 242) D
      '#FFC801',  # (255, 202, 1) E
      '#581AF2',  # (88, 26, 242) F
      '#1AF295',  # (26, 242, 149) G
      '#1A95F2',  # (26, 149, 242) H
      '#F2761A',  # (242, 118, 26) I
      '#1A39F2',  # (26, 57, 242) J
      '#F21A1A',  # (242, 26, 26) K
      '#F21AD3',  # (242, 26, 211) L
      '#B4F21A',  # (180, 242, 26) M
      '#1AF2F2',  # (26, 242, 242) N
    ]
    
    # Map specifically to letters A-N as requested
    chapter_color_map = dict(zip(list('ABCDEFGHIJKLMN'), hex_colors))
    
    chapter_colors = {ch: chapter_color_map.get(ch, '#CCCCCC') for ch in chapters}

    wide = (counts
        .pivot(index="age_num", columns="ICD_Chapter", values="count")
        .reindex(index=ages, columns=chapters)
        .fillna(0.0))

    # -----------------------------
    # X positions + labels
    # -----------------------------
    group_spacing = 1.6
    x = np.arange(len(ages)) * group_spacing
    
    # Age labels
    age_labels = ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70-79']
    
    # robust label mapping
    if min(ages) == 1 and max(ages) <= len(age_labels):
        age_label_map = {a: age_labels[a - 1] for a in ages}
    else:
        age_label_map = {a: str(a) for a in ages}

    # -----------------------------
    # Layout parameters (tune)
    # -----------------------------
    gap = 2.0          # vertical gap between ribbons WITHIN each age column (in "count units")
    curviness = 0.55   # 0..1, higher = curvier
    alpha = 0.90
    min_height = 0.0   # set >0 to drop tiny ribbons

    # -----------------------------
    # Compute stacked intervals per age
    # - "largest on top" at each age
    # - anchored at 0 (NO centering)
    # -----------------------------
    y0 = {a: {} for a in ages}
    y1 = {a: {} for a in ages}
    column_heights = {}

    for a in ages:
        row = wide.loc[a].to_dict()

        # order chapters by count desc (largest will end on top)
        ordered = sorted(chapters, key=lambda ch: row[ch], reverse=True)

        cum = 0.0
        nonzero = 0
        for ch in ordered[::-1]:  # stack bottom->top
            h = float(row[ch])
            if h <= 0:
                y0[a][ch] = cum
                y1[a][ch] = cum
                continue

            y0[a][ch] = cum
            y1[a][ch] = cum + h
            cum += h
            nonzero += 1

            # add gap after each nonzero stratum except the last one (handled below)
            cum += gap

        # remove trailing gap so total height matches: sum(counts) + gap*(k-1)
        if nonzero > 0:
            cum -= gap

        column_heights[a] = cum
    
    # Find max height for plot limits
    max_height = max(column_heights.values()) if column_heights else 1.0

    # -----------------------------
    # Build and draw ribbons
    # -----------------------------
    # Using local figsize to match snippet provided (2.00, 2.20)
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(2.00, 2.20))
    
    ribbons = []
    for i in range(len(ages) - 1):
        a0, a1 = ages[i], ages[i + 1]
        x0, x1 = x[i], x[i + 1]

        for ch in chapters:
            h0 = y1[a0][ch] - y0[a0][ch]
            h1 = y1[a1][ch] - y0[a1][ch]
            if max(h0, h1) <= min_height:
                continue

            path = ribbon_path(
                x0, y0[a0][ch], y1[a0][ch],
                x1, y0[a1][ch], y1[a1][ch],
                bend=curviness
            )
            ribbons.append((max(h0, h1), ch, path))

    # draw thin -> thick so big ones sit on top
    ribbons.sort(key=lambda t: t[0])

    for _, ch, path in ribbons:
        ax.add_patch(PathPatch(path, facecolor=chapter_colors[ch], edgecolor="none", alpha=alpha))

    # -----------------------------
    # Axes / labels
    # -----------------------------
    ax.set_xlim(x.min() - group_spacing * 0.6, x.max() + group_spacing * 0.6)
    
    # Set ylim slightly above max height to fit
    ax.set_ylim(0, max_height + 5 if max_height > 35 else 40)
    
    ax.set_xticks(x)
    ax.set_xticklabels([age_label_map[a] for a in ages], rotation=45, ha="right")
    ax.set_xlabel("Age group")
    ax.set_ylabel("n mortality sinks")
    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(x.min(), x.max())
    ax.spines["left"].set_bounds(0, max_height)

    # Optional: Legend (commented out in snippet, but good to have logic ready)
    # handles = [Line2D([0], [0], color=chapter_colors[ch], lw=8) for ch in chapters]
    # ax.legend(handles, [f"Chapter {ch}" for ch in chapters],
    #           title="ICD Chapter", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(fig, output_name)
    plt.close(fig)

@app.command()
def main():
    logger.info("Generating Panel Plots...")
    
    # --- Plot 1: Mortality Sinks ("The Plot of Before") ---
    sinks_path = PROCESSED_DATA_DIR / 'high_mortality_sinks_ZSCORE.csv'
    if sinks_path.exists():
        logger.info("Plotting Mortality Sinks...")
        df_sinks = pd.read_csv(sinks_path)
        # Sinks file contains only sinks, so we plot all rows
        generate_bar_panel(df_sinks, 'panel_sinks_by_age_sex', 'n mortality sinks')
        
        # --- Plot 4: Alluvial Sinks (New Request) ---
        generate_alluvial_plot(df_sinks, "panel_alluvial_curved_gapped_anchored_by_age_icd_chapter")
    else:
        logger.warning(f"Missing {sinks_path}. Skipping Sinks plot.")

    # --- Plot 2: High-Degree Outliers ("This One") ---
    # Prefer 'Outliers_EXACT.csv' as it aligns with robustness analysis, fall back to 'outliers_data_FINAL.csv'
    outliers_path = PROCESSED_DATA_DIR / 'Outliers_EXACT.csv'
    if not outliers_path.exists():
        outliers_path = PROCESSED_DATA_DIR / 'outliers_data_FINAL.csv'

    if outliers_path.exists():
        logger.info("Plotting High-Degree Outliers...")
        df_outliers = pd.read_csv(outliers_path)
        
        # Filter for Hubs: Outlier flag must be True AND Deviation > 0 (High Degree)
        # Check column names carefully as they might differ slightly between files
        col_map = {c.lower(): c for c in df_outliers.columns}
        outlier_col = col_map.get('outlier')
        dev_col = col_map.get('deviation')

        if outlier_col and dev_col:
            df_hubs = df_outliers[
                (df_outliers[outlier_col] == True) & 
                (df_outliers[dev_col] > 0)
            ].copy()
            
            generate_bar_panel(df_hubs, 'panel_outliers_by_age_sex', 'n high-degree outliers')
        else:
            logger.error(f"Missing 'Outlier' or 'Deviation' columns in {outliers_path}.")
    else:
        logger.warning("Missing Outliers data file. Run 'tapas outliers' first.")

    # --- Plot 3: Bridge Edges (New Request) ---
    bridge_path = PROCESSED_DATA_DIR / 'bridge_edges_mortality_ZSCORE.csv'
    if bridge_path.exists():
        logger.info("Plotting Bridge Edges...")
        df_bridge = pd.read_csv(bridge_path)
        # Bridge file contains filtered edges, plot all
        generate_bar_panel(df_bridge, 'panel_bridge_edges_by_age_sex', 'n bridge edges')
    else:
        logger.warning(f"Missing {bridge_path}. Skipping Bridge Edges plot.")

if __name__ == "__main__":
    app()