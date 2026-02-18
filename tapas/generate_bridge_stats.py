"""
Generate Bridge Edges with Mortality and Z-Score statistics.

This script implements the methodology to identify 'Low-to-High Mortality Bridges':
1. Calculates Edge Betweenness Centrality for each Sex/Age network.
2. Integrates Mortality data (loading from sex-specific files).
3. Computes Z-Scores for Betweenness and Mortality Difference.
4. Filters for critical bridges based on the paper's thresholds:
   - Mortality Gap >= 30%
   - Top 5% of Z-score product
"""

import numpy as np
import pandas as pd
from pathlib import Path
import typer
from loguru import logger
from typing import Dict, List, Tuple, Optional

# Import project configuration and features
from tapas.config import (
    DATA_DIR, 
    PROCESSED_DATA_DIR, 
    INTERIM_DATA_DIR,
    AGE_GROUPS, 
    SEXES
)
from tapas.features import NetworkAnalyzer

app = typer.Typer()

def load_node_mapping(data_dir: Path) -> List[str]:
    """
    Load the list of ICD codes that corresponds to the matrix indices.
    Priority:
    1. ICD10_Diagnoses_All.csv (Standard)
    2. mortality_diag_Female.csv (Fallback: assumes same order as matrix)
    """
    # 1. Try Standard Mapping File
    mapping_path = data_dir / "ICD10_Diagnoses_All.csv"
    
    # 2. Fallback to Female mortality file if main map is missing
    if not mapping_path.exists():
        fallback_path = data_dir / "mortality_diag_Female.csv"
        if fallback_path.exists():
            logger.info(f"Mapping file not found. Using {fallback_path.name} for node list.")
            mapping_path = fallback_path
        else:
            logger.error(f"Node mapping file not found at {mapping_path}")
            raise FileNotFoundError(f"Missing {mapping_path}")
        
    df = pd.read_csv(mapping_path)
    
    # Robustly find the ICD column
    possible_cols = ["ICD", "Code", "Diagnosis", "icd"]
    for col in possible_cols:
        if col in df.columns:
            return df[col].astype(str).tolist()
    
    # Fallback: assume first column if it looks like ICD codes (strings)
    if pd.api.types.is_object_dtype(df.iloc[:, 0]) or pd.api.types.is_string_dtype(df.iloc[:, 0]):
         return df.iloc[:, 0].astype(str).tolist()

    return df.iloc[:, 0].astype(str).tolist()

def load_mortality_data(data_dir: Path, sex: str) -> pd.DataFrame:
    """
    Load mortality rates for a specific sex.
    Looks for: mortality_diag_{sex}.csv (e.g., mortality_diag_Female.csv)
    """
    # Try the specific file format provided by user
    filename = f"mortality_diag_{sex}.csv"
    path = data_dir / filename
    
    if not path.exists():
        # Try generic fallback just in case
        path = data_dir / "ICD10_Diagnoses_All.csv"
        
    if not path.exists():
        logger.error(f"Mortality file not found for {sex}. Expected: {filename}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    return df

def find_column_fuzzy(df: pd.DataFrame, keywords: List[str]) -> str:
    """Helper to find a column name matching all keywords case-insensitively."""
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

def get_mortality_rate(df_mort: pd.DataFrame, icd: str, age_idx: int, age_label: str) -> float:
    """
    Retrieve mortality rate for a specific condition from a Sex-Specific DataFrame.
    """
    if df_mort.empty:
        return 0.0

    # Potential column names for Age Group
    candidates = [
        f"age_{age_idx}",       # age_1
        str(age_idx),           # 1
        f"Mortality_{age_idx}", # Mortality_1
        age_label,              # 0-9
        f"age_{age_label}",     # age_0-9
        f"Mortality_{age_label}"# Mortality_0-9
    ]
    
    col_name = None
    for cand in candidates:
        if cand in df_mort.columns:
            col_name = cand
            break
            
    # Fuzzy match if exact failed
    if not col_name:
        col_name = find_column_fuzzy(df_mort, [str(age_idx)])
        
    if not col_name or col_name not in df_mort.columns:
        return 0.0
        
    # Find ICD column
    icd_col = None
    possible_cols = ["ICD", "Code", "Diagnosis", "icd"]
    for c in possible_cols:
        if c in df_mort.columns:
            icd_col = c
            break
    if not icd_col: 
        icd_col = df_mort.columns[0] # Fallback
        
    row = df_mort[df_mort[icd_col] == icd]
    if row.empty:
        return 0.0
    
    val = row[col_name].values[0]
    
    # CLEANING: Handle string percentages or NaNs
    try:
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            val = val.replace('%', '').strip()
            if '<' in val:
                val = val.replace('<', '')
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def z_score(series: pd.Series) -> pd.Series:
    """Compute Z-score avoiding division by zero."""
    if len(series) < 2:
        return series * 0
    std = series.std()
    if std == 0:
        return series * 0
    return (series - series.mean()) / std

def add_english_descriptions(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Add English descriptions to the DataFrame using ICD10_Diagnoses_All.csv.
    """
    desc_path = data_dir / "ICD10_Diagnoses_All.csv"
    
    if not desc_path.exists():
        # Fallback to female mortality file which might have descriptions
        desc_path = data_dir / "mortality_diag_Female.csv"

    if not desc_path.exists():
        logger.warning(f"Description file not found at {desc_path}. Returning without descriptions.")
        return df

    try:
        # Load description file
        desc_df = pd.read_csv(desc_path)
        
        # Identify ICD and Description columns
        icd_col = None
        desc_col = None
        
        for col in desc_df.columns:
            if col.lower() in ['icd', 'code', 'diagnosis_code']:
                icd_col = col
            if col.lower() in ['description', 'name', 'diagnosis', 'long_description']:
                desc_col = col
        
        # Fallback if columns not explicitly named
        if not icd_col: icd_col = desc_df.columns[0]
        if not desc_col and len(desc_df.columns) > 1: desc_col = desc_df.columns[1]

        if icd_col and desc_col:
            # Create mapping dict
            desc_map = dict(zip(desc_df[icd_col].astype(str), desc_df[desc_col]))
            
            # Map Source and Target
            df['source_desc'] = df['source'].map(desc_map).fillna("Unknown")
            df['target_desc'] = df['target'].map(desc_map).fillna("Unknown")
            logger.info("Added English descriptions.")
        else:
            logger.warning("Could not identify ICD/Description columns.")
            
    except Exception as e:
        logger.warning(f"Failed to add descriptions: {e}")

    return df

@app.command()
def main(
    output_filename: str = "bridge_edges_mortality_ZSCORE.csv"
):
    """
    Main execution loop to generate bridge statistics.
    """
    # Define base paths - robust check similar to other_tables.py
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        logger.info(f"Path {base_data_path} not found, falling back to {DATA_DIR}")
        base_data_path = DATA_DIR

    # Load Node List (Mapping)
    try:
        node_list = load_node_mapping(base_data_path)
        logger.info(f"Loaded {len(node_list)} nodes from mapping.")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    all_bridges = []
    
    for sex in SEXES:
        # Load Mortality Data SPECIFIC to this Sex
        df_mortality = load_mortality_data(base_data_path, sex)
        if df_mortality.empty:
            logger.warning(f"Skipping {sex} - No mortality data found.")
            continue
            
        logger.info(f"Loaded mortality data for {sex}. Columns: {list(df_mortality.columns)}")

        for age_idx, age_label in AGE_GROUPS.items():
            
            logger.info(f"Processing {sex} - Age Group {age_label} (Index {age_idx})...")
            
            # 1. Load Network
            adj_file = base_data_path / "3.AdjacencyMatrices" / f"Adj_Matrix_{sex}_ICD_age_{age_idx}.csv"
            
            if not adj_file.exists():
                logger.warning(f"Adjacency file not found: {adj_file}")
                continue
                
            try:
                G = NetworkAnalyzer.load_adjacency_matrix(adj_file)
                analyzer = NetworkAnalyzer(G)
            except Exception as e:
                logger.error(f"Failed to load network for {sex} {age_label}: {e}")
                continue

            # 2. Calculate Edge Betweenness
            try:
                edge_betweenness = analyzer.get_edge_betweenness(G, normalized=True)
            except Exception as e:
                logger.error(f"Betweenness calc failed: {e}")
                continue
            
            if not edge_betweenness:
                logger.warning("No edges found in network.")
                continue

            # 3. Process Edges
            group_edges = []
            
            for (u, v), bet_val in edge_betweenness.items():
                if u >= len(node_list) or v >= len(node_list):
                    continue
                    
                icd_u = node_list[u]
                icd_v = node_list[v]
                
                # Get rates from the Sex-Specific DataFrame
                mort_u = get_mortality_rate(df_mortality, icd_u, age_idx, age_label)
                mort_v = get_mortality_rate(df_mortality, icd_v, age_idx, age_label)
                
                # Determine Low vs High
                if mort_u < mort_v:
                    source, target = icd_u, icd_v
                    mort_source, mort_target = mort_u, mort_v
                else:
                    source, target = icd_v, icd_u
                    mort_source, mort_target = mort_v, mort_u
                    
                mort_diff = mort_target - mort_source
                
                # Filter A: Paper threshold is 30% (0.3)
                if mort_diff < 0.3:
                    continue
                
                group_edges.append({
                    "sex": sex,
                    "age_group": age_label,
                    "age_idx": age_idx,
                    "source": source,
                    "target": target,
                    "edge_betweenness": bet_val,
                    "mortality_source": mort_source,
                    "mortality_target": mort_target,
                    "mortality_diff": mort_diff
                })
            
            if not group_edges:
                logger.debug(f"No edges with mortality diff >= 0.3 found for {sex} {age_label}")
                continue
                
            # 4. Compute Z-Scores for this group
            df_group = pd.DataFrame(group_edges)
            
            df_group["z_betweenness"] = z_score(df_group["edge_betweenness"])
            df_group["z_mortality_diff"] = z_score(df_group["mortality_diff"])
            df_group["z_score_product"] = df_group["z_betweenness"] * df_group["z_mortality_diff"]
            
            # 5. Filter B: Top 5% of Z-score product
            if len(df_group) > 0:
                threshold_z = df_group["z_score_product"].quantile(0.95)
                filtered_group = df_group[df_group["z_score_product"] >= threshold_z]
                
                logger.info(f"Found {len(filtered_group)} bridge edges for {sex} {age_label}")
                all_bridges.append(filtered_group)

    # Combine all
    if all_bridges:
        final_df = pd.concat(all_bridges, ignore_index=True)
        
        # Add English Descriptions (Feature from other_tables.py)
        final_df = add_english_descriptions(final_df, base_data_path)
        
        # Save
        output_path = PROCESSED_DATA_DIR / output_filename
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        logger.success(f"Generated bridge statistics with {len(final_df)} edges.")
        logger.success(f"Saved to {output_path}")
    else:
        logger.warning("No bridges found meeting criteria. Check column names in mortality files.")

if __name__ == "__main__":
    app()