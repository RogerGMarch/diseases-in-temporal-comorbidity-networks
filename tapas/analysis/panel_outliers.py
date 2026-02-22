"""
Panel Visualization: Outliers, Sinks, and Bridges.

Generates panel figures comparing counts by Age and Sex:
1. High-Degree Outliers (Hubs) - Bar chart.
2. High-Mortality Sinks - Bar chart.
3. Bridge Edges - Bar chart.
4. Mortality Sinks Alluvial Flow - Ribbon plot showing ICD Chapter distribution across ages.
5. Disease by Age Heatmap - Heatmap of high-degree outliers colored by ICD Chapter.
6. Outlier Scatter Panels - Degree vs Prevalence scatter plots per age group (Figure 2 / Fig. S2).

All plots share specific visual styles (pastel colors, dimensions, spine adjustments).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from adjustText import adjust_text
import typer
from loguru import logger
from pathlib import Path
from typing import Optional, List, Dict

from tapas.config import PROCESSED_DATA_DIR, FIGURES_DIR, DATA_DIR, AGE_GROUPS, SEXES

app = typer.Typer()

# Set style parameters globally
plt.rcParams['font.family'] = 'Ubuntu'
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



# -------------------------------------------------------------------------
# Bridge Edge Plots (from notebook 010)
# -------------------------------------------------------------------------

# Age labels used across all bridge plots
_AGE_LABELS_FULL = ['0 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79']
_MIN_AGE_NUM = 5  # Bridge edges only start from age group 5

_HEX_COLORS = [
    '#1AF239', '#58F21A', '#961D1A', '#B41AF2', '#FFC801', '#581AF2',
    '#1AF295', '#1A95F2', '#F2761A', '#1A39F2', '#F21A1A', '#F21AD3',
    '#B4F21A', '#1AF2F2',
]
_CHAPTER_COLOR_MAP = dict(zip(list('ABCDEFGHIJKLMN'), _HEX_COLORS))


def _prepare_bridge_df(df_bridge: pd.DataFrame, df_sinks: pd.DataFrame) -> pd.DataFrame:
    """
    Shared preparation for all bridge plots:
    - Normalises age column to age_num
    - Filters to age groups >= _MIN_AGE_NUM
    - Derives ICD chapter columns, Chapter_Pair, Lower_Mort_Chapter
    - Builds chapter_colors from the sinks file (matching notebook 009 palette)
    Returns (df_edges, chapter_colors, age_ticks).
    """
    col_map = {c.lower(): c for c in df_bridge.columns}
    age_col = col_map.get('age_group') or 'Age_Group'

    df = df_bridge.copy()
    if df[age_col].dtype == object:
        df['age_num'] = df[age_col].astype(str).str.replace('age_', '', regex=False).astype(int)
    else:
        df['age_num'] = df[age_col].astype(int)

    df = df[df['age_num'] >= _MIN_AGE_NUM].copy()

    # ICD chapter columns
    df['ICD_Chapter_1'] = df['ICD_Code_1'].astype(str).str[0]
    df['ICD_Chapter_2'] = df['ICD_Code_2'].astype(str).str[0]

    def lower_mort_chapter(r):
        m1 = r.get('Mortality_1')
        m2 = r.get('Mortality_2')
        if pd.isna(m1) and pd.isna(m2):
            return str(r['ICD_Chapter_1'])
        if pd.isna(m2) or (pd.notna(m1) and m1 <= m2):
            return str(r['ICD_Chapter_1'])
        return str(r['ICD_Chapter_2'])

    df['Lower_Mort_Chapter'] = df.apply(lower_mort_chapter, axis=1)

    df['Chapter_Pair'] = df.apply(
        lambda r: '-'.join(sorted([str(r['ICD_Chapter_1']), str(r['ICD_Chapter_2'])])),
        axis=1
    )

    # Chapter colors — derive unique chapters from sinks file to match 009 palette
    if df_sinks is not None and 'ICD_Code' in df_sinks.columns:
        outlier_chapters = df_sinks['ICD_Code'].astype(str).str[0]
        unique_chapters = sorted(outlier_chapters.unique())
    else:
        unique_chapters = sorted(pd.unique(pd.concat([
            df['ICD_Code_1'].astype(str).str[0],
            df['ICD_Code_2'].astype(str).str[0]
        ])))

    chapter_colors = {ch: _CHAPTER_COLOR_MAP.get(ch, '#CCCCCC') for ch in unique_chapters}

    return df, chapter_colors


def generate_bridge_bar_panel(df_bridge: pd.DataFrame, df_sinks: pd.DataFrame, output_name: str):
    """
    Bar chart of bridge edge counts by age group and sex.
    """
    logger.info(f"Generating Bridge Bar Panel: {output_name}...")

    col_map = {c.lower(): c for c in df_bridge.columns}
    sex_col = col_map.get('sex') or col_map.get('gender') or 'Sex'

    df_edges, chapter_colors = _prepare_bridge_df(df_bridge, df_sinks)[:2]

    counts = (
        df_edges.groupby([sex_col, 'age_num'])
        .size()
        .reset_index(name='count')
    )

    if counts.empty:
        logger.warning(f"No data for {output_name}")
        return

    fig, ax = plt.subplots(figsize=(2.00, 1.16))

    colors = {'Male': '#1FA3FF', 'Female': '#FF5A8A'}
    for sex, sub in counts.groupby(sex_col):
        color = colors.get(str(sex), '#CCCCCC')
        offset = 0.15 if str(sex).lower().startswith('m') else -0.15
        ax.bar(sub['age_num'] + offset, sub['count'], width=0.3, label=str(sex), color=color)

    age_ticks = np.arange(_MIN_AGE_NUM, len(_AGE_LABELS_FULL) + 1)
    age_labels = _AGE_LABELS_FULL[_MIN_AGE_NUM - 1:]
    ax.set_xticks(age_ticks)
    ax.set_xticklabels(age_labels, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Age', fontsize=8)
    ax.set_ylabel('n bridge edges', fontsize=8)

    ax.set_xlim(age_ticks.min() - 0.5, age_ticks.max() + 0.5)
    ax.set_ylim(0, counts['count'].max() + 2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(age_ticks.min(), age_ticks.max())
    ax.spines['left'].set_bounds(0, counts['count'].max() + 2)

    save_figure(fig, output_name)
    plt.close(fig)


def generate_bridge_alluvial(df_bridge: pd.DataFrame, df_sinks: pd.DataFrame, output_name: str):
    """
    Alluvial ribbon plot of bridge edges by age group, colored by lower-mortality ICD chapter pair.
    Matches notebook 010 Cell 6.
    """
    logger.info(f"Generating Bridge Alluvial: {output_name}...")

    df_edges, chapter_colors = _prepare_bridge_df(df_bridge, df_sinks)[:2]

    pair_counts = (
        df_edges.groupby(['age_num', 'Chapter_Pair'])
        .size()
        .reset_index(name='count')
    )

    ages  = sorted(df_edges['age_num'].unique())
    pairs = sorted(pair_counts['Chapter_Pair'].unique())

    pair_to_low_chapter = (
        df_edges.groupby('Chapter_Pair')['Lower_Mort_Chapter']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    pair_colors = {p: chapter_colors.get(pair_to_low_chapter.get(p, ''), '#CCCCCC') for p in pairs}

    wide = (pair_counts
            .pivot(index='age_num', columns='Chapter_Pair', values='count')
            .reindex(index=ages, columns=pairs)
            .fillna(0.0))

    group_spacing = 1.6
    x = np.arange(len(ages)) * group_spacing
    age_label_map = {a: _AGE_LABELS_FULL[a - 1] for a in ages}

    gap       = 2.0
    curviness = 0.55
    alpha     = 0.90

    y0 = {a: {} for a in ages}
    y1 = {a: {} for a in ages}
    column_heights = {}

    for a in ages:
        row = wide.loc[a].to_dict()
        ordered = sorted(pairs, key=lambda p: row[p], reverse=True)
        cum, nonzero = 0.0, 0
        for p in ordered[::-1]:
            h = float(row[p])
            if h <= 0:
                y0[a][p] = y1[a][p] = cum
                continue
            y0[a][p] = cum
            y1[a][p] = cum + h
            cum += h + gap
            nonzero += 1
        if nonzero > 0:
            cum -= gap
        column_heights[a] = cum

    fig, ax = plt.subplots(figsize=(2.00, 2.20))

    ribbons = []
    for i in range(len(ages) - 1):
        a0, a1 = ages[i], ages[i + 1]
        x0, x1 = x[i], x[i + 1]
        for p in pairs:
            h0 = y1[a0][p] - y0[a0][p]
            h1 = y1[a1][p] - y0[a1][p]
            if max(h0, h1) <= 0:
                continue
            path = ribbon_path(
                x0, y0[a0][p], y1[a0][p],
                x1, y0[a1][p], y1[a1][p],
                bend=curviness
            )
            ribbons.append((max(h0, h1), p, path))

    ribbons.sort(key=lambda t: t[0])
    for _, p, path in ribbons:
        ax.add_patch(PathPatch(path, facecolor=pair_colors[p], edgecolor='none', alpha=alpha))

    ax.set_xlim(x.min() - group_spacing * 0.6, x.max() + group_spacing * 0.6)
    ax.set_ylim(0, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([age_label_map[a] for a in ages], rotation=45, ha='right')
    ax.set_xlabel('Age group')
    ax.set_ylabel('n bridges')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(x.min(), x.max())
    ax.spines['left'].set_bounds(0, 60)

    save_figure(fig, output_name)
    plt.close(fig)

    # Legend: chapter pairs
    _generate_bridge_pair_legend(pairs, pair_colors)


def generate_bridge_heatmap(df_bridge: pd.DataFrame, df_sinks: pd.DataFrame, output_name: str):
    """
    Heatmap of bridge edges (disease couples) by age group.
    Rows = low-to-high mortality edge labels. Color = lower-mortality ICD chapter.
    Uses pcolormesh for crisp vector PDF output.
    Matches notebook 010 Cell 7.
    """
    logger.info(f"Generating Bridge Heatmap: {output_name}...")

    def short_icd(icd):
        if not icd or pd.isna(icd):
            return 'NA'
        return str(icd).split('.', 1)[0]

    df_edges, chapter_colors = _prepare_bridge_df(df_bridge, df_sinks)[:2]

    # Build low-to-high edge label
    df_tmp = df_edges.copy()
    df_tmp['icd1_short'] = df_tmp['ICD_Code_1'].map(short_icd)
    df_tmp['icd2_short'] = df_tmp['ICD_Code_2'].map(short_icd)

    def edge_label_low_high(r):
        m1 = r.get('Mortality_1')
        m2 = r.get('Mortality_2')
        i1, i2 = str(r['icd1_short']), str(r['icd2_short'])
        if pd.isna(m1) and pd.isna(m2):
            return '-'.join(sorted([i1, i2]))
        if pd.isna(m2) or (pd.notna(m1) and m1 <= m2):
            return f"{i1}-{i2}"
        return f"{i2}-{i1}"

    df_tmp['Edge_Label'] = df_tmp.apply(edge_label_low_high, axis=1)

    edge_to_low_chapter = (
        df_tmp.drop_duplicates('Edge_Label')
              .set_index('Edge_Label')['Lower_Mort_Chapter']
              .to_dict()
    )

    # Presence/absence matrix
    df_unique = df_tmp[['Edge_Label', 'age_num']].drop_duplicates()
    heatmap_data = (
        df_unique.assign(present=1)
        .groupby(['Edge_Label', 'age_num'])['present']
        .max()
        .unstack(fill_value=0)
    )

    age_ticks = np.array(sorted(df_edges['age_num'].unique()))
    heatmap_data = heatmap_data.reindex(columns=age_ticks, fill_value=0)

    # Sort rows by first age of appearance
    present       = heatmap_data.values > 0
    first_age_idx = present.argmax(axis=1)
    first_age_idx[~present.any(axis=1)] = heatmap_data.shape[1]
    sorted_idx    = np.argsort(first_age_idx)
    heatmap_data_sorted = heatmap_data.iloc[sorted_idx]

    edge_labels        = heatmap_data_sorted.index.tolist()
    n_edges, n_ages    = heatmap_data_sorted.shape

    present   = heatmap_data_sorted.values > 0
    color_idx = np.tile(np.arange(n_edges)[:, None], (1, n_ages))
    masked_idx = np.ma.masked_where(~present, color_idx)

    row_colors = [chapter_colors.get(edge_to_low_chapter.get(lbl, ''), '#CCCCCC') for lbl in edge_labels]
    cmap = mcolors.ListedColormap(row_colors)
    cmap.set_bad(color='white')

    # Use pcolormesh for vector PDF (no blurring)
    plot_data = masked_idx.astype(float)
    plot_data[masked_idx.mask] = np.nan

    fig_edge, ax_edge = plt.subplots(figsize=(2.08, 5.8))
    ax_edge.pcolormesh(
        plot_data,
        cmap=cmap,
        vmin=0,
        vmax=n_edges - 1,
        shading='flat',
        linewidth=0,
        antialiased=False,
    )
    ax_edge.invert_yaxis()

    x_tick_labels = [_AGE_LABELS_FULL[a - 1] for a in age_ticks]
    ax_edge.set_xticks(np.arange(n_ages) + 0.5)
    ax_edge.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=8)
    ax_edge.set_yticks(np.arange(n_edges) + 0.5)
    ax_edge.set_yticklabels(edge_labels, fontsize=8)
    ax_edge.set_xlabel('Age group', fontsize=8)
    ax_edge.set_ylabel('Disease couples', fontsize=8)

    ax_edge.spines['top'].set_visible(False)
    ax_edge.spines['right'].set_visible(False)

    save_figure(fig_edge, output_name)
    plt.close(fig_edge)


def _generate_bridge_pair_legend(pairs, pair_colors):
    """
    Legend-only figure for bridge chapter pairs (two half-columns).
    Matches notebook 010 Cell 8.
    """
    fig, ax = plt.subplots(figsize=(2.8, 1.76))
    ax.axis('off')

    legend_handles = [plt.Line2D([0], [0], color=pair_colors[p], lw=6) for p in pairs]
    legend_labels  = list(pairs)

    mid = (len(legend_handles) + 1) // 2
    h1, l1 = legend_handles[:mid], legend_labels[:mid]
    h2, l2 = legend_handles[mid:], legend_labels[mid:]

    leg1 = ax.legend(h1, l1, title='Chapter Pair', frameon=False,
                     loc='center left', bbox_to_anchor=(0.0, 0.5))
    ax.add_artist(leg1)
    ax.legend(h2, l2, title='Chapter Pair', frameon=False,
              loc='center left', bbox_to_anchor=(0.5, 0.5))

    fig.tight_layout()
    save_figure(fig, 'legend_bridge_edges_heatmap_chapter_pair')
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 2 / Fig. S2: Outlier Scatter Panels (Degree vs Prevalence)
# -------------------------------------------------------------------------

def _load_prevalence_degree_data(lower_q: float = 0.05, upper_q: float = 0.95) -> pd.DataFrame:
    """
    Load the precomputed prevalence-degree analysis CSV and recompute
    ``quintile_category`` using the requested percentile thresholds
    (default 5th / 95th, ~10 % of nodes highlighted per group).

    The pre-baked columns ``log_ratio_20th_percentile`` /
    ``log_ratio_80th_percentile`` are replaced with fresh per-group values
    so the histogram dashed lines stay consistent with the colouring.
    """
    csv_path = PROCESSED_DATA_DIR / "prevalence_degree_analysis.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run the prevalence-degree analysis first "
            "(python pipeline.py)."
        )
    df = pd.read_csv(csv_path)

    # Recompute thresholds and category per sex/age group
    rows = []
    for (sex, ag), grp in df.groupby(["sex", "age_group"], sort=False):
        grp = grp.copy()
        p_low  = grp["log_ratio"].quantile(lower_q)
        p_high = grp["log_ratio"].quantile(upper_q)
        grp["log_ratio_20th_percentile"] = p_low   # reuse column name for histogram
        grp["log_ratio_80th_percentile"] = p_high
        grp["is_low_quintile"]  = grp["log_ratio"] <= p_low
        grp["is_high_quintile"] = grp["log_ratio"] >= p_high
        grp["quintile_category"] = "middle"
        grp.loc[grp["is_low_quintile"],  "quintile_category"] = "low"
        grp.loc[grp["is_high_quintile"], "quintile_category"] = "high"
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


def generate_outlier_scatter_panel(
    sex: str = "Female",
    n_labels: int = 6,
    output_name: Optional[str] = None,
) -> None:
    """
    Generate the 4x2 panel figure of Degree vs Prevalence (log-log) for a
    given sex, with one subplot per age group.

    Each subplot consists of:
      - A main scatter area (log-log) where every disease is plotted as a dot
        colored by ICD chapter (A-N).  Outliers (quintile_category != 'middle')
        are drawn as solid, borderless markers; non-outliers are
        small and semi-transparent.
      - A narrow right histogram showing the distribution of
        log10(Degree / Prevalence) with dashed lines at the 20th/80th
        percentile thresholds. Y-axis ticks are on the left (base of bars).

    A shared legend of ICD Disease Categories is placed below the panels.

    Parameters
    ----------
    sex : str
        ``"Female"`` (default, Figure 2) or ``"Male"`` (Fig. S2).
    n_labels : int
        Number of outlier ICD codes to label per panel (highest |log_ratio|
        deviation from the group median).
    output_name : str or None
        Base name for the saved file.  Defaults to
        ``figure2_outlier_scatter_female`` /
        ``figS2_outlier_scatter_male``.
    """
    logger.info(f"Generating Outlier Scatter Panels for {sex}...")

    # -- Load data --
    df_all = _load_prevalence_degree_data()
    df = df_all[df_all["sex"] == sex].copy()

    if df.empty:
        logger.warning(f"No prevalence-degree data for sex={sex}. Skipping.")
        return

    df["icd_chapter"] = df["icd_code"].astype(str).str[0]
    age_groups_ordered = [AGE_GROUPS[k] for k in sorted(AGE_GROUPS.keys())]

    if output_name is None:
        output_name = "figure2_outlier_scatter_female" if sex == "Female" else "figS2_outlier_scatter_male"

    # -- Compute global axis limits for consistent comparison across subplots --
    all_prev = df["prevalence"].replace(0, np.nan).dropna()
    all_deg  = df["degree"].replace(0, np.nan).dropna()
    all_lr   = df["log_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

    global_xlim = (all_prev.min() * 0.7, all_prev.max() * 1.4)
    global_ylim = (all_deg.min()  * 0.7, all_deg.max()  * 1.4)
    global_hist_ylim = (all_lr.min() - 0.1 * all_lr.std(),
                        all_lr.max() + 0.1 * all_lr.std())

    # -- Figure layout: 4 rows x 2 cols --
    fig = plt.figure(figsize=(4.8, 9.0))
    outer_gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        wspace=0.48,
        hspace=0.52,
        bottom=0.07,
        top=0.96,
        left=0.09,
        right=0.97,
    )

    for idx, age_label in enumerate(age_groups_ordered):
        row, col = divmod(idx, 2)

        # Inner grid: wide scatter + narrow histogram (flush, ticks on histogram left)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec=outer_gs[row, col],
            width_ratios=[5, 1],
            wspace=0.0,
        )
        ax_scatter = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1])

        sub = df[df["age_group"] == age_label].copy()

        if sub.empty:
            ax_scatter.text(0.5, 0.5, "No data", transform=ax_scatter.transAxes,
                            ha="center", va="center", fontsize=7, color="#999999")
            ax_hist.axis("off")
            continue

        degree     = sub["degree"].values
        prevalence = sub["prevalence"].values
        log_ratio  = sub["log_ratio"].values
        chapters   = sub["icd_chapter"].values
        categories = sub["quintile_category"].values

        is_outlier = categories != "middle"   # ~40% per group — 20th/80th pct boundary
        is_normal  = ~is_outlier

        # Non-outliers: small, semi-transparent
        for ch in sorted(set(chapters)):
            ch_color = _CHAPTER_COLOR_MAP.get(ch, "#CCCCCC")
            mask = is_normal & (chapters == ch)
            if mask.any():
                ax_scatter.scatter(prevalence[mask], degree[mask],
                                   s=8, c=ch_color, alpha=0.55,
                                   edgecolors="none", zorder=1, rasterized=True)

        # Outliers: larger, fully opaque, dark circle edge
        for ch in sorted(set(chapters)):
            ch_color = _CHAPTER_COLOR_MAP.get(ch, "#CCCCCC")
            mask = is_outlier & (chapters == ch)
            if mask.any():
                ax_scatter.scatter(prevalence[mask], degree[mask],
                                   s=22, c=ch_color, alpha=1.0,
                                   edgecolors="#333333", linewidths=0.3, zorder=3)

        # Labels: top 5 outliers by |log_ratio deviation from group median|
        if is_outlier.any():
            outlier_sub = sub[is_outlier].copy()
            outlier_sub["_abs_dev"] = (outlier_sub["log_ratio"] - sub["log_ratio"].median()).abs()
            label_sub = outlier_sub.nlargest(5, "_abs_dev")
        else:
            label_sub = pd.DataFrame()

        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlim(global_xlim)
        ax_scatter.set_ylim(global_ylim)
        ax_scatter.set_title(f"Age Group {age_label}", fontsize=8, pad=3, loc="left")
        ax_scatter.tick_params(labelsize=6, which="both")
        ax_scatter.set_ylabel("Degree (log scale)" if col == 0 else "", fontsize=7)
        ax_scatter.set_xlabel("Prevalence (log scale)" if row == 3 else "", fontsize=7)
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        # Non-overlapping ICD labels — done after axes scales are set so
        # adjust_text can use real data-coordinate limits
        texts = []
        for _, r in label_sub.iterrows():
            texts.append(
                ax_scatter.text(r["prevalence"], r["degree"], r["icd_code"],
                                fontsize=5, fontweight="bold", alpha=0.95,
                                clip_on=True)
            )
        if texts:
            adjust_text(
                texts,
                ax=ax_scatter,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5,
                                shrinkA=2, shrinkB=2),
                expand=(1.2, 1.4),
                force_text=(0.3, 0.5),
                force_points=(0.1, 0.2),
                only_move={"points": "xy", "texts": "xy", "objects": "xy"},
            )

        # Right histogram: y-axis on left (base of bars)
        valid_lr = log_ratio[np.isfinite(log_ratio)]
        if len(valid_lr) > 0:
            ax_hist.hist(valid_lr, bins=25, orientation="horizontal",
                         color="#7F8C8D", alpha=0.55, edgecolor="grey", linewidth=0.3)
            p20 = sub["log_ratio_20th_percentile"].iloc[0]
            p80 = sub["log_ratio_80th_percentile"].iloc[0]
            ax_hist.axhline(p20, color="#3498DB", ls="--", lw=0.7, alpha=0.8)
            ax_hist.axhline(p80, color="#E74C3C", ls="--", lw=0.7, alpha=0.8)
        ax_hist.set_ylim(global_hist_ylim)

        ax_hist.yaxis.set_label_position("left")
        ax_hist.yaxis.tick_left()
        ax_hist.tick_params(axis="y", labelsize=5, which="both")
        ax_hist.set_xticks([])
        ax_hist.set_ylabel("")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["bottom"].set_visible(False)

    # Shared ICD chapter legend
    all_chapters = sorted(df["icd_chapter"].unique())
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_CHAPTER_COLOR_MAP.get(ch, "#CCCCCC"),
               markeredgecolor="none", markersize=6, label=ch)
        for ch in all_chapters
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(all_chapters),
               frameon=False, fontsize=7, title="ICD Disease Categories",
               title_fontsize=7, handletextpad=0.3, columnspacing=0.8,
               bbox_to_anchor=(0.5, 0.0))

    save_figure(fig, output_name)
    plt.close(fig)
    logger.success(f"Saved outlier scatter panel: {output_name}")


def generate_all_outlier_scatter_panels() -> None:
    """Generate outlier scatter panels for both Female (Fig. 2) and Male (Fig. S2)."""
    for sex in SEXES:
        try:
            generate_outlier_scatter_panel(sex=sex)
        except FileNotFoundError as e:
            logger.warning(str(e))
        except Exception as e:
            logger.error(f"Failed to generate outlier scatter panel for {sex}: {e}")
            raise


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

    # --- Plot 3 + Bridge plots (from notebook 010) ---
    bridge_path = PROCESSED_DATA_DIR / 'bridge_edges_mortality_ZSCORE.csv'
    if bridge_path.exists():
        logger.info("Plotting Bridge Edges...")
        df_bridge = pd.read_csv(bridge_path)

        # df_sinks used to derive chapter_colors consistent with 009 palette
        _df_sinks = pd.read_csv(sinks_path) if sinks_path.exists() else None

        # Bar chart: bridge edges by age & sex
        generate_bridge_bar_panel(df_bridge, _df_sinks, 'panel_bridge_edges_by_age_sex')

        # Alluvial: bridge edges by age, colored by chapter pair
        generate_bridge_alluvial(df_bridge, _df_sinks, 'panel_bridge_edges_alluvial_by_age_chapter_pair')

        # Heatmap: disease couples by age (vector PDF via pcolormesh)
        generate_bridge_heatmap(df_bridge, _df_sinks, 'panel_bridge_edges_by_age_heatmap_short_icd_edges')
    else:
        logger.warning(f"Missing {bridge_path}. Skipping Bridge Edges plots.")

    # --- Figure 2 / Fig. S2: Outlier Scatter Panels ---
    logger.info("Generating Outlier Scatter Panels (Figure 2 + Fig. S2)...")
    generate_all_outlier_scatter_panels()


if __name__ == "__main__":
    app()