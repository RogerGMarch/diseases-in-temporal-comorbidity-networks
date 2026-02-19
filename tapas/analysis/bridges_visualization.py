"""
Visualization script for mortality differences and z-scores.

This script implements the methodology from '004_MortalityEdges_Distributions.ipynb' to:
1. Load all edge data across demographic groups.
2. Generate histograms for Mortality Differences to aid threshold selection.
3. Generate distributions for Z-scores (Betweenness, Mortality Diff, and Product).
4. Print detailed statistics on edge counts at various thresholds.

It uses `features.py` for robust data loading.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from tapas.config import PROCESSED_DATA_DIR, FIGURES_DIR, SEXES, AGE_GROUPS
from tapas.features import NetworkAnalyzer

app = typer.Typer()


def load_all_edges_data() -> pd.DataFrame:
    """
    Load pre-computed edge metrics for all sex/age groups using NetworkAnalyzer.
    """
    all_data = []
    
    logger.info("Loading edge data for all groups...")
    
    for sex in SEXES:
        for age_id, age_range in tqdm(AGE_GROUPS.items(), desc=f"Loading {sex} data"):
            # NetworkAnalyzer.load_edge_metrics calculates betweenness and maps mortality
            # Returns DataFrame with columns: 
            # Sex, Age_Group, Edge_Betweenness, Mortality_Diff, Mortality_1, Mortality_2, etc.
            df = NetworkAnalyzer.load_edge_metrics(sex, age_id)
            
            if not df.empty:
                all_data.append(df)
            else:
                logger.debug(f"No data for {sex} Age {age_id}")
    
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)


def plot_mortality_difference_histograms(df: pd.DataFrame, output_dir: Path):
    """
    Plot histograms of mortality differences (Overall, Zoomed, By Sex, Cumulative).
    """
    # Use a style that handles grid lines well
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mortality Difference Distributions', fontsize=16, fontweight='bold')
    
    # 1. Overall distribution
    ax = axes[0, 0]
    ax.hist(df['Mortality_Diff'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15% threshold')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20% threshold')
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('All Edges (Overall Distribution)', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 2. Zoomed in (0-0.3 range)
    ax = axes[0, 1]
    df_subset = df[df['Mortality_Diff'] <= 0.3]
    ax.hist(df_subset['Mortality_Diff'], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10%')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15%')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20%')
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Zoomed: 0-30% Range', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 3. By sex
    ax = axes[1, 0]
    for sex in SEXES:
        sex_data = df[df['Sex'] == sex]['Mortality_Diff']
        ax.hist(sex_data, bins=80, alpha=0.6, label=sex, edgecolor='black')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('By Sex', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_diffs = np.sort(df['Mortality_Diff'])
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    ax.plot(sorted_diffs, cumulative, linewidth=2, color='navy')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.axvline(0.15, color='orange', linestyle='--', linewidth=2, label='15% threshold')
    ax.axvline(0.20, color='green', linestyle='--', linewidth=2, label='20% threshold')
    
    # Add percentage labels
    for threshold in [0.10, 0.15, 0.20]:
        pct = (df['Mortality_Diff'] >= threshold).mean() * 100
        # Find y-position on the curve (roughly) or just use the calculated mean
        y_pos = (df['Mortality_Diff'] < threshold).mean() # CDF value
        ax.text(threshold + 0.01, y_pos, f'{pct:.1f}% above', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Mortality Difference', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'mortality_difference_histograms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved: {output_path}")
    plt.close()


def plot_zscore_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Calculate and plot Z-score distributions (Betweenness, Mort Diff, Product).
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Z-Score Distributions for Threshold Selection', fontsize=16, fontweight='bold')
    
    # Calculate z-scores per sex-age group
    all_zscores = []
    
    for sex in SEXES:
        for age_group in df['Age_Group'].unique():
            subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_group)].copy()
            
            if len(subset) == 0:
                continue
            
            # Z-scores for betweenness
            bet_mean = subset['Edge_Betweenness'].mean()
            bet_std = subset['Edge_Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Edge_Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            # Z-scores for mortality diff
            mort_mean = subset['Mortality_Diff'].mean()
            mort_std = subset['Mortality_Diff'].std()
            if mort_std > 0:
                subset['z_mort_diff'] = (subset['Mortality_Diff'] - mort_mean) / mort_std
            else:
                subset['z_mort_diff'] = 0
            
            # Z-score product
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            
            all_zscores.append(subset)
    
    df_z = pd.concat(all_zscores, ignore_index=True)
    
    percentiles = [90, 95, 99]

    # 1. Z-score betweenness distribution
    ax = axes[0, 0]
    ax.hist(df_z['z_betweenness'], bins=100, edgecolor='black', alpha=0.7, color='coral')
    for p in percentiles:
        val = df_z['z_betweenness'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Betweenness Z-Scores', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 2. Z-score mortality diff distribution
    ax = axes[0, 1]
    ax.hist(df_z['z_mort_diff'], bins=100, edgecolor='black', alpha=0.7, color='lightgreen')
    for p in percentiles:
        val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Mortality Difference Z-Scores', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 3. Z-score product distribution
    ax = axes[0, 2]
    # Only positive products
    positive_products = df_z[df_z['z_product'] > 0]['z_product']
    ax.hist(positive_products, bins=100, edgecolor='black', alpha=0.7, color='skyblue')
    for p in [90, 95, 99]:
        val = positive_products.quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}th %ile')
    ax.set_xlabel('Z-Score Product', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Z-Score Product (Positive Only)', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 4. Cumulative distribution - betweenness
    ax = axes[1, 0]
    sorted_z = np.sort(df_z['z_betweenness'])
    cumulative = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    ax.plot(sorted_z, cumulative, linewidth=2, color='coral')
    for p in percentiles:
        val = df_z['z_betweenness'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}%')
        # pct_above = (df_z['z_betweenness'] >= val).mean() * 100
        # ax.text(val + 0.1, 0.5, f'{100-p}% above', fontsize=9,
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative: Betweenness', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 5. Cumulative distribution - mortality diff
    ax = axes[1, 1]
    sorted_z = np.sort(df_z['z_mort_diff'])
    cumulative = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    ax.plot(sorted_z, cumulative, linewidth=2, color='lightgreen')
    for p in percentiles:
        val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(val, linestyle='--', linewidth=2, label=f'{p}%')
    ax.set_xlabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative: Mortality Diff', fontsize=13, fontweight='bold')
    ax.legend()
    
    # 6. Scatter plot: betweenness vs mortality diff z-scores
    ax = axes[1, 2]
    scatter = ax.scatter(df_z['z_betweenness'], df_z['z_mort_diff'], 
                        c=df_z['z_product'], cmap='RdYlGn', alpha=0.3, s=1)
    
    # Add percentile lines
    for p in [90, 95]:
        bet_val = df_z['z_betweenness'].quantile(p/100)
        mort_val = df_z['z_mort_diff'].quantile(p/100)
        ax.axvline(bet_val, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(mort_val, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Z-Score (Betweenness)', fontsize=12)
    ax.set_ylabel('Z-Score (Mortality Diff)', fontsize=12)
    ax.set_title('Z-Score Scatter (color = product)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Z-Product')
    
    plt.tight_layout()
    output_path = output_dir / 'zscore_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"✓ Saved: {output_path}")
    plt.close()


def print_threshold_statistics(df: pd.DataFrame):
    """Print statistics for different thresholds."""
    
    logger.info("\n" + "="*80)
    logger.info("MORTALITY DIFFERENCE THRESHOLD STATISTICS")
    logger.info("="*80)
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"\n{'Threshold':>12} | {'# Edges':>10} | {'% of Total':>12} | {'Female':>8} | {'Male':>8}")
    print("-" * 80)
    
    total = len(df)
    for threshold in thresholds:
        count = (df['Mortality_Diff'] >= threshold).sum()
        pct = count / total * 100
        female_count = ((df['Mortality_Diff'] >= threshold) & (df['Sex'] == 'Female')).sum()
        male_count = ((df['Mortality_Diff'] >= threshold) & (df['Sex'] == 'Male')).sum()
        
        print(f"{threshold:11.0%} | {count:10,} | {pct:11.1f}% | {female_count:8,} | {male_count:8,}")
    
    logger.info("\n" + "="*80)
    logger.info("Z-SCORE PERCENTILE STATISTICS (Method 2)")
    logger.info("="*80)
    
    # Re-calculate z-scores locally for statistics (same logic as plotting)
    all_zscores = []
    for sex in SEXES:
        for age_group in df['Age_Group'].unique():
            subset = df[(df['Sex'] == sex) & (df['Age_Group'] == age_group)].copy()
            if len(subset) == 0:
                continue
            
            bet_mean = subset['Edge_Betweenness'].mean()
            bet_std = subset['Edge_Betweenness'].std()
            if bet_std > 0:
                subset['z_betweenness'] = (subset['Edge_Betweenness'] - bet_mean) / bet_std
            else:
                subset['z_betweenness'] = 0
            
            mort_mean = subset['Mortality_Diff'].mean()
            mort_std = subset['Mortality_Diff'].std()
            if mort_std > 0:
                subset['z_mort_diff'] = (subset['Mortality_Diff'] - mort_mean) / mort_std
            else:
                subset['z_mort_diff'] = 0
            
            subset['z_product'] = subset['z_betweenness'] * subset['z_mort_diff']
            all_zscores.append(subset)
    
    if not all_zscores:
        logger.warning("No data available for Z-score statistics.")
        return

    df_z = pd.concat(all_zscores, ignore_index=True)
    
    # Filter for positive z-products and minimum mortality diff (as per notebook)
    df_z_filtered = df_z[(df_z['z_product'] > 0) & (df_z['Mortality_Diff'] >= 0.10)]
    
    percentiles = [80, 85, 90, 95, 99]
    
    print(f"\n{'Percentile':>12} | {'# Edges':>10} | {'% of Filtered':>15} | {'Female':>8} | {'Male':>8}")
    print("-" * 80)
    
    for p in percentiles:
        threshold_val = df_z_filtered['z_product'].quantile(p/100)
        count = (df_z_filtered['z_product'] >= threshold_val).sum()
        pct = count / len(df_z_filtered) * 100
        female_count = ((df_z_filtered['z_product'] >= threshold_val) & (df_z_filtered['Sex'] == 'Female')).sum()
        male_count = ((df_z_filtered['z_product'] >= threshold_val) & (df_z_filtered['Sex'] == 'Male')).sum()
        
        print(f"{p:11}th | {count:10,} | {pct:14.1f}% | {female_count:8,} | {male_count:8,}")


@app.command()
def main():
    """Main execution function."""
    
    print("="*80)
    print("THRESHOLD SELECTION VISUALIZATION")
    print("="*80)
    
    print("\nLoading all edge data...")
    df = load_all_edges_data()
    
    if df.empty:
        logger.error("No edge data found. Check data availability.")
        return

    print(f"Total edges: {len(df):,}")
    
    # Ensure output directory exists (using FIGURES_DIR now)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating mortality difference histograms...")
    plot_mortality_difference_histograms(df, FIGURES_DIR)
    
    print("\nGenerating z-score distributions...")
    plot_zscore_distributions(df, FIGURES_DIR)
    
    print_threshold_statistics(df)
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {FIGURES_DIR / 'mortality_difference_histograms.png'}")
    print(f"  - {FIGURES_DIR / 'zscore_distributions.png'}")

if __name__ == '__main__':
    app()