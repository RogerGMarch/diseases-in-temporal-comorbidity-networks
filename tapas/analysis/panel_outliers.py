"""
Panel Visualization: Outliers, Sinks, and Bridges.

Generates panel figures comparing counts by Age and Sex:
1. High-Degree Outliers (Hubs) - Bar chart.
2. High-Mortality Sinks - Bar chart.
3. Bridge Edges - Bar chart.
4. Mortality Sinks Alluvial Flow - Ribbon plot showing ICD Chapter distribution across ages.
5. Disease by Age Heatmap - Heatmap of high-degree outliers colored by ICD Chapter.

All plots share specific visual styles (pastel colors, dimensions, spine adjustments).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    ax.legend(frameon=False, fontsize=8, title_fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.15))

    max_count = counts['count'].max()
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, max_count + 2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_bounds(1, 8)
    ax.spines['left'].set_bounds(0, max_count + 2)

    save_figure(fig, output_name)
    plt.close(fig)


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

    df = df_in.copy()

    col_map = {c.lower(): c for c in df.columns}
    icd_col = col_map.get('icd_code')

    if 'Age_Group' in df.columns:
        df['age_num'] = df['Age_Group'].astype(int)
    elif 'age_num' not in df.columns:
        logger.error("Could not determine 'age_num' for alluvial plot.")
        return

    if icd_col:
        df["ICD_Chapter"] = df[icd_col].astype(str).str[0]
    else:
        logger.error("Could not find ICD Code column for alluvial plot.")
        return

    counts = (
        df.groupby(["age_num", "ICD_Chapter"])
          .size()
          .reset_index(name="count"))

    ages = sorted(counts["age_num"].unique())
    chapters = sorted(counts["ICD_Chapter"].unique())

    hex_colors = [
        '#1AF239', '#58F21A', '#961D1A', '#B41AF2', '#FFC801', '#581AF2',
        '#1AF295', '#1A95F2', '#F2761A', '#1A39F2', '#F21A1A', '#F21AD3',
        '#B4F21A', '#1AF2F2',
    ]
    chapter_color_map = dict(zip(list('ABCDEFGHIJKLMN'), hex_colors))
    chapter_colors = {ch: chapter_color_map.get(ch, '#CCCCCC') for ch in chapters}

    wide = (counts
            .pivot(index="age_num", columns="ICD_Chapter", values="count")
            .reindex(index=ages, columns=chapters)
            .fillna(0.0))

    group_spacing = 1.6
    x = np.arange(len(ages)) * group_spacing

    age_labels = ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70-79']

    if min(ages) == 1 and max(ages) <= len(age_labels):
        age_label_map = {a: age_labels[a - 1] for a in ages}
    else:
        age_label_map = {a: str(a) for a in ages}

    gap = 2.0
    curviness = 0.55
    alpha = 0.90
    min_height = 0.0

    y0 = {a: {} for a in ages}
    y1 = {a: {} for a in ages}
    column_heights = {}

    for a in ages:
        row = wide.loc[a].to_dict()
        ordered = sorted(chapters, key=lambda ch: row[ch], reverse=True)

        cum = 0.0
        nonzero = 0
        for ch in ordered[::-1]:
            h = float(row[ch])
            if h <= 0:
                y0[a][ch] = cum
                y1[a][ch] = cum
                continue

            y0[a][ch] = cum
            y1[a][ch] = cum + h
            cum += h
            nonzero += 1
            cum += gap

        if nonzero > 0:
            cum -= gap

        column_heights[a] = cum

    max_height = max(column_heights.values()) if column_heights else 1.0

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

    ribbons.sort(key=lambda t: t[0])

    for _, ch, path in ribbons:
        ax.add_patch(PathPatch(path, facecolor=chapter_colors[ch], edgecolor="none", alpha=alpha))

    ax.set_xlim(x.min() - group_spacing * 0.6, x.max() + group_spacing * 0.6)
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

    save_figure(fig, output_name)
    plt.close(fig)


# -------------------------------------------------------------------------
# Heatmap
# -------------------------------------------------------------------------

def generate_disease_age_heatmap(df_in: pd.DataFrame, output_name: str):
    """
    Generates a heatmap of diseases by age, colored by ICD chapter.

    Matches the notebook exactly:
    - Input must be df_sinks (high_mortality_sinks_ZSCORE.csv) with NO extra filtering.
    - Presence/absence only (deduplicates to unique Description_Eng x age_num pairs).
    - Sorts rows by first age of appearance.
    - Colors by ICD Chapter via masked array + ListedColormap.
    """
    logger.info(f"Generating Heatmap: {output_name}...")

    df = df_in.copy()

    # --- 1. Normalize column names ---
    col_map = {c.lower(): c for c in df.columns}
    desc_col = col_map.get('description_eng') or col_map.get('description') or 'Description_Eng'
    icd_col  = col_map.get('icd_code') or col_map.get('icd') or 'ICD_Code'

    # Ensure age_num exists
    if 'age_num' not in df.columns:
        age_col = col_map.get('age_group') or 'Age_Group'
        if df[age_col].dtype == object:
            df['age_num'] = (
                df[age_col].astype(str)
                .str.replace(r'(?i)age_', '', regex=True)
            )
            df['age_num'] = pd.to_numeric(df['age_num'], errors='coerce').fillna(0).astype(int)
        else:
            df['age_num'] = df[age_col].fillna(0).astype(int)

    df['Description_Eng'] = df[desc_col]
    df['ICD_Code']        = df[icd_col]

    if df.empty:
        logger.warning("No data to plot for heatmap.")
        return

    # --- 2. Map disease -> ICD code (first occurrence) ---
    desc_to_icd: Dict[str, str] = {}
    for d, icd in zip(df['Description_Eng'], df['ICD_Code']):
        if pd.notna(d) and pd.notna(icd):
            desc_to_icd[str(d)] = str(icd)

    def icd_to_chapter(icd):
        if not icd or pd.isna(icd):
            return 'Unknown'
        return str(icd)[0]

    def short_icd(icd):
        if not icd or pd.isna(icd):
            return 'NA'
        return str(icd).split('.', 1)[0]

    # --- 3. Build presence/absence matrix (deduplicated) ---
    df_unique = df[['Description_Eng', 'age_num']].drop_duplicates()

    heatmap_data = (
        df_unique.assign(present=1)
        .groupby(['Description_Eng', 'age_num'])['present']
        .max()
        .unstack(fill_value=0)
    )

    age_labels = ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70-79']
    age_ticks  = np.arange(1, len(age_labels) + 1)
    heatmap_data = heatmap_data.reindex(columns=age_ticks, fill_value=0)

    # --- 4. Sort rows by first age of appearance ---
    present       = heatmap_data.values > 0
    first_age_idx = present.argmax(axis=1)
    first_age_idx[~present.any(axis=1)] = heatmap_data.shape[1]
    sorted_idx    = np.argsort(first_age_idx)
    heatmap_data_sorted = heatmap_data.iloc[sorted_idx]

    disease_names      = heatmap_data_sorted.index.tolist()
    n_diseases, n_ages = heatmap_data_sorted.shape

    # --- 5. Build masked colour index (one unique colour per disease row) ---
    present    = heatmap_data_sorted.values > 0
    color_idx  = np.tile(np.arange(n_diseases)[:, None], (1, n_ages))
    masked_idx = np.ma.masked_where(~present, color_idx)

    # --- 6. ICD chapter colours ---
    chapters    = []
    short_codes = []
    for name in disease_names:
        icd = desc_to_icd.get(name, None)
        chapters.append(icd_to_chapter(icd))
        short_codes.append(short_icd(icd))

    unique_chapters = sorted(set(chapters))
    hex_colors = [
        '#1AF239', '#58F21A', '#961D1A', '#B41AF2', '#FFC801', '#581AF2',
        '#1AF295', '#1A95F2', '#F2761A', '#1A39F2', '#F21A1A', '#F21AD3',
        '#B4F21A', '#1AF2F2',
    ]
    chapter_color_map = dict(zip(list('ABCDEFGHIJKLMN'), hex_colors))
    chapter_colors    = {ch: chapter_color_map.get(ch, '#CCCCCC') for ch in unique_chapters}

    row_colors = [chapter_colors.get(ch, '#CCCCCC') for ch in chapters]
    cmap = mcolors.ListedColormap(row_colors)
    cmap.set_bad(color='white')

    # --- 7. Plot ---
    # Use pcolormesh instead of imshow so the PDF is fully vector (no pixel blurring).
    fig_disease, ax_disease = plt.subplots(figsize=(1.80, 4.80))

    # pcolormesh expects a plain numpy array, not a masked array — set absent cells to NaN
    plot_data = masked_idx.astype(float)
    plot_data[masked_idx.mask] = np.nan

    # Build a version of the colormap that maps NaN to white
    cmap.set_bad(color='white')

    ax_disease.pcolormesh(
        plot_data,
        cmap=cmap,
        vmin=0,
        vmax=n_diseases - 1,
        shading='flat',       # one colour per cell, no interpolation
        linewidth=0,
        antialiased=False,
    )

    # pcolormesh y-axis is bottom-up by default; flip so row 0 is at the top
    ax_disease.invert_yaxis()

    # Tick positions sit at cell centres (0.5, 1.5, …)
    ax_disease.set_xticks(np.arange(n_ages) + 0.5)
    ax_disease.set_xticklabels(age_labels, rotation=45, fontsize=8, ha='right')
    ax_disease.set_yticks(np.arange(n_diseases) + 0.5)
    ax_disease.set_yticklabels(short_codes, fontsize=8)
    ax_disease.set_xlabel('Age group', fontsize=8)
    ax_disease.set_ylabel('Disease', fontsize=8)

    ax_disease.spines['top'].set_visible(False)
    ax_disease.spines['right'].set_visible(False)

    save_figure(fig_disease, output_name)
    plt.close(fig_disease)

    # --- 8. Legend: ICD Chapters (single column) ---
    fig_leg1, ax_leg1 = plt.subplots(figsize=(3.20, 2.20))
    ax_leg1.axis('off')
    legend_handles = [plt.Line2D([0], [0], color=chapter_colors[ch], lw=6) for ch in unique_chapters]
    legend_labels  = [f'Chapter {ch}' for ch in unique_chapters]
    ax_leg1.legend(legend_handles, legend_labels, title='ICD Chapter', frameon=False, loc='center')
    save_figure(fig_leg1, 'legend_disease_by_age_heatmap_icd_chapter')
    plt.close(fig_leg1)

    # --- 9. Legend: ICD Chapters (three columns) ---
    fig_leg2, ax_leg2 = plt.subplots(figsize=(4.40, 2.40))
    ax_leg2.axis('off')
    ax_leg2.legend(legend_handles, legend_labels, title='ICD Chapters', frameon=False,
                   loc='center', ncol=3, columnspacing=1.2, handlelength=1.6)
    fig_leg2.tight_layout()
    save_figure(fig_leg2, 'legend_disease_by_age_heatmap_icd_chapter_3col')
    plt.close(fig_leg2)

    # --- 10. Legend: Sex ---
    fig_leg3, ax_leg3 = plt.subplots(figsize=(2.60, 1.40))
    ax_leg3.axis('off')
    sex_legend_handles = [plt.Line2D([0], [0], color=SEX_COLORS[s], lw=6) for s in ['Male', 'Female']]
    ax_leg3.legend(sex_legend_handles, ['Male', 'Female'], title='Sex', frameon=False,
                   loc='center', ncol=2, columnspacing=1.2, handlelength=1.6)
    fig_leg3.tight_layout()
    save_figure(fig_leg3, 'legend_sex')
    plt.close(fig_leg3)


@app.command()
def main():
    logger.info("Generating Panel Plots...")

    # --- Plot 1 + 4 + 5: Mortality Sinks ---
    sinks_path = PROCESSED_DATA_DIR / 'high_mortality_sinks_ZSCORE.csv'
    if sinks_path.exists():
        logger.info("Plotting Mortality Sinks...")
        df_sinks = pd.read_csv(sinks_path)

        # Plot 1: Bar chart
        generate_bar_panel(df_sinks, 'panel_sinks_by_age_sex', 'n mortality sinks')

        # Plot 4: Alluvial
        generate_alluvial_plot(df_sinks, "panel_alluvial_curved_gapped_anchored_by_age_icd_chapter")

        # Plot 5: Disease heatmap — uses df_sinks directly, NO extra filtering
        generate_disease_age_heatmap(df_sinks, 'panel_disease_by_age_heatmap')

    else:
        logger.warning(f"Missing {sinks_path}. Skipping Sinks plots.")

    # --- Plot 2: High-Degree Outliers ---
    outliers_path = PROCESSED_DATA_DIR / 'Outliers_EXACT.csv'
    if not outliers_path.exists():
        outliers_path = PROCESSED_DATA_DIR / 'outliers_data_FINAL.csv'

    if outliers_path.exists():
        logger.info("Plotting High-Degree Outliers...")
        df_outliers = pd.read_csv(outliers_path)

        col_map     = {c.lower(): c for c in df_outliers.columns}
        outlier_col = col_map.get('outlier')
        dev_col     = col_map.get('deviation')

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

    # --- Plot 3: Bridge Edges ---
    bridge_path = PROCESSED_DATA_DIR / 'bridge_edges_mortality_ZSCORE.csv'
    if bridge_path.exists():
        logger.info("Plotting Bridge Edges...")
        df_bridge = pd.read_csv(bridge_path)
        generate_bar_panel(df_bridge, 'panel_bridge_edges_by_age_sex', 'n bridge edges')
    else:
        logger.warning(f"Missing {bridge_path}. Skipping Bridge Edges plot.")


if __name__ == "__main__":
    app()