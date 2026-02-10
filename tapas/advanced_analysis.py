"""Advanced network analysis: prevalence-degree correlation and high-risk disease identification."""

from pathlib import Path
from typing import Dict, Optional

from loguru import logger
import numpy as np
import pandas as pd

from tapas.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from tapas.features import NetworkAnalyzer
from tapas.network_analysis import AGE_GROUPS, SEXES, get_adjacency_matrix_path

# Age group mapping: age_1 = 0-9, age_2 = 10-19, ..., age_8 = 70-79
AGE_GROUP_MAP = {v: k for k, v in AGE_GROUPS.items()}

# Reverse mapping: age_10 (from mortality data) to age group string
AGE_10_TO_GROUP = {
    1: "0-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70-79",
}


def get_node_icd_mapping(sex: str, age_num: int) -> Dict[int, str]:
    """
    Extract node index to ICD code mapping from GEXF file.

    Args:
        sex: "Female" or "Male"
        age_num: Age group number (1-8)

    Returns:
        Dictionary mapping node index (0-based) to ICD code
    """
    import xml.etree.ElementTree as ET

    graphs_dir = INTERIM_DATA_DIR / "extracted" / "Data" / "4.Graphs-gexffiles"
    gexf_file = graphs_dir / f"Graph_{sex}_ICD_Age_{age_num}.gexf"

    if not gexf_file.exists():
        logger.warning(f"GEXF file not found: {gexf_file}")
        return {}

    try:
        tree = ET.parse(gexf_file)
        root = tree.getroot()
        ns = {"gexf": "http://www.gexf.net/1.3"}

        mapping = {}
        nodes = root.findall(".//gexf:node", ns)
        for node in nodes:
            # GEXF node IDs are 1-indexed, adjacency matrix is 0-indexed
            node_id = int(node.get("id", 0)) - 1
            node_label = node.get("label", "")
            if node_label:
                mapping[node_id] = node_label

        return mapping
    except Exception as e:
        logger.warning(f"Error extracting node mapping from {gexf_file}: {e}")
        return {}


def load_prevalence_data(prevalence_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load prevalence data from CSV file.

    Args:
        prevalence_path: Path to prevalence CSV file. If None, uses default processed path.

    Returns:
        DataFrame with prevalence data
    """
    if prevalence_path is None:
        prevalence_path = PROCESSED_DATA_DIR / "prevalence_data.csv"

    if not prevalence_path.exists():
        raise FileNotFoundError(f"Prevalence data not found: {prevalence_path}")

    df = pd.read_csv(prevalence_path)
    logger.info(f"Loaded prevalence data: {len(df)} rows")
    return df


def get_prevalence_for_stratum(
    prevalence_df: pd.DataFrame, sex: str, age_group: str, year: int = 2014
) -> pd.Series:
    """
    Get prevalence values for a specific sex-age group-year combination.

    Args:
        prevalence_df: DataFrame with prevalence data
        sex: "Female" or "Male"
        age_group: Age group string (e.g., "0-9", "10-19")
        year: Year to use (default: 2014, most recent in dataset)

    Returns:
        Series with ICD codes as index and prevalence as values
    """
    mask = (
        (prevalence_df["sex"] == sex)
        & (prevalence_df["Age_Group"] == age_group)
        & (prevalence_df["year"] == year)
    )
    stratum_data = prevalence_df[mask].copy()
    return pd.Series(
        stratum_data["p"].values, index=stratum_data["icd_code"].values, name="prevalence"
    )


def compute_log_ratio(degree: float, prevalence: float) -> Optional[float]:
    """
    Compute log-ratio: log10(degree / prevalence).

    The paper uses log base 10, not natural log.

    Args:
        degree: Node degree
        prevalence: Disease prevalence

    Returns:
        Log-ratio value (log10), or None if degree <= 0 or prevalence <= 0
    """
    if degree > 0 and prevalence > 0:
        return np.log10(degree / prevalence)
    return None


def analyze_prevalence_degree_correlation(
    sex: str,
    age_group: str,
    prevalence_df: pd.DataFrame,
    year: int = 2014,
) -> pd.DataFrame:
    """
    Analyze prevalence-degree correlation for a specific sex-age group.

    Args:
        sex: "Female" or "Male"
        age_group: Age group string (e.g., "0-9")
        prevalence_df: DataFrame with prevalence data
        year: Year to use for prevalence (default: 2014)

    Returns:
        DataFrame with ICD codes, degree, prevalence, log_ratio, and quintile flags
    """
    # Load graph
    age_num = AGE_GROUP_MAP[age_group]
    adj_path = get_adjacency_matrix_path(sex, age_num)

    if not adj_path.exists():
        logger.warning(f"Adjacency matrix not found: {adj_path}")
        return pd.DataFrame()

    G = NetworkAnalyzer.load_adjacency_matrix(adj_path)

    # Create analyzer instance to reuse calculations
    analyzer = NetworkAnalyzer(G)
    G_filtered = analyzer.get_filtered_graph(G)

    # Get prevalence data for this stratum
    prevalence_series = get_prevalence_for_stratum(prevalence_df, sex, age_group, year)

    # Get node-to-ICD mapping from GEXF file
    node_mapping = get_node_icd_mapping(sex, age_num)

    # Get node degrees from filtered graph
    degrees = dict(G_filtered.degree())

    results = []

    for node_idx, degree in degrees.items():
        # Map node index to ICD code
        icd_code = node_mapping.get(node_idx, str(node_idx))

        # Get prevalence for this ICD code
        prevalence = prevalence_series.get(icd_code, 0.0)

        if prevalence > 0:
            log_ratio = compute_log_ratio(degree, prevalence)
            if log_ratio is not None:
                results.append(
                    {
                        "icd_code": icd_code,
                        "degree": degree,
                        "prevalence": prevalence,
                        "log_ratio": log_ratio,
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute quintiles
    if len(df) > 0:
        df["log_ratio_20th_percentile"] = df["log_ratio"].quantile(0.20)
        df["log_ratio_80th_percentile"] = df["log_ratio"].quantile(0.80)
        df["is_low_quintile"] = df["log_ratio"] <= df["log_ratio_20th_percentile"]
        df["is_high_quintile"] = df["log_ratio"] >= df["log_ratio_80th_percentile"]
        df["quintile_category"] = "middle"
        df.loc[df["is_low_quintile"], "quintile_category"] = "low"
        df.loc[df["is_high_quintile"], "quintile_category"] = "high"

    return df


def compute_z_score(values: pd.Series) -> pd.Series:
    """
    Compute Z-scores for a series of values.

    Args:
        values: Series of numeric values

    Returns:
        Series of Z-scores
    """
    mean = values.mean()
    std = values.std()
    if std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def identify_high_risk_nodes(
    sex: str,
    age_group: str,
    mortality_df: pd.DataFrame,
    top_percentile: float = 0.20,
) -> pd.DataFrame:
    """
    Identify high-risk nodes (high betweenness centrality and high mortality).

    This implements the "high-mortality sinks" analysis from the paper:
    - Nodes with high betweenness centrality (central in network structure)
    - AND high in-hospital mortality rates
    - Uses product of z-scores: z(betweenness) × z(mortality)
    - Selects top 20% of nodes by z-product

    Methodology (as per reference implementation and paper):
    1. Include ALL connected nodes: left-merge mortality (mortality=0 when missing), then
       compute z-scores on this full stratum so mean/std are over all diagnoses in the network
    2. Compute z_product = z_betweenness × z_mortality (logical AND - high only if both are high)
    3. Filter to nodes with BOTH z_betweenness > 0 AND z_mortality > 0
       (diagnosis must be above average in both dimensions - very strict definition)
    4. Calculate geometric mean: sqrt(z_betweenness × z_mortality) for ranking/reporting
    5. Select top X% (default 20%) by z_product within each sex × age group:
       keep nodes with z_product >= (100 - X)th percentile of z_product among
       positive-positive nodes (percentile threshold, not fixed count)

    The geometric mean is used for ranking/interpretability but not for selection.
    Selection is based on z_product, which penalizes diagnoses that are:
    - High mortality but peripheral (low betweenness)
    - Central but low mortality

    These nodes are particularly harmful as they lie on many shortest paths
    (high betweenness) and have high mortality rates.

    Args:
        sex: "Female" or "Male"
        age_group: Age group string (e.g., "0-9")
        mortality_df: DataFrame with columns: icd_code, mortality (and optionally sex, age_group)
        top_percentile: Top percentile to select (default: 0.20 for top 20%)

    Returns:
        DataFrame with high-risk nodes, including:
        - z_betweenness, z_mortality, z_product (for selection)
        - z_geom_mean (for ranking/reporting)
    """
    # Load graph and compute betweenness
    age_num = AGE_GROUP_MAP[age_group]
    adj_path = get_adjacency_matrix_path(sex, age_num)

    if not adj_path.exists():
        logger.warning(f"Adjacency matrix not found: {adj_path}")
        return pd.DataFrame()

    G = NetworkAnalyzer.load_adjacency_matrix(adj_path)

    # Create analyzer instance to reuse calculations
    analyzer = NetworkAnalyzer(G)

    # Get node-to-ICD mapping
    node_mapping = get_node_icd_mapping(sex, age_num)

    # Compute betweenness centrality on FULL graph (normalized for individual node analysis)
    # Reuse NetworkAnalyzer method which calculates on full graph and caches results
    betweenness = analyzer.get_node_betweenness(G, normalized=True, use_cache=True)

    # Filter to only connected nodes (nodes with at least one neighbor) for reporting
    G_filtered = analyzer.get_filtered_graph(G)

    # Filter to only connected nodes (nodes with at least one neighbor)
    # The paper states: "We filtered for nodes with at least one neighbor"
    filtered_nodes = set(G_filtered.nodes())

    # Prepare betweenness data with ICD codes for connected nodes only
    # Script 2 calculates z-scores on all connected nodes in the sex-age group subset
    betweenness_df = pd.DataFrame(
        [
            {
                "icd_code": node_mapping.get(node, str(node)),
                "betweenness": bc,
            }
            for node, bc in betweenness.items()
            if node in filtered_nodes  # Only connected nodes
        ]
    )

    # Filter mortality data for this stratum
    if "sex" in mortality_df.columns and "age_group" in mortality_df.columns:
        mortality_stratum = mortality_df[
            (mortality_df["sex"] == sex) & (mortality_df["age_group"] == age_group)
        ].copy()
    else:
        mortality_stratum = mortality_df.copy()

    # Merge betweenness and mortality: LEFT merge so we include ALL connected nodes.
    # Nodes not in mortality data get mortality=0 (manuscript: "attach mortality" to each node).
    # Z-scores are then computed on this full stratum so mean/std are over all diagnoses
    # in the network, not just those with recorded mortality.
    merged = betweenness_df.merge(
        mortality_stratum[["icd_code", "mortality"]], on="icd_code", how="left"
    )
    merged["mortality"] = merged["mortality"].fillna(0.0)

    if len(merged) == 0:
        logger.warning(f"No matching nodes found for {sex} {age_group}")
        return pd.DataFrame()

    # Compute Z-scores on ALL connected nodes (full stratum per manuscript).
    # This is critical: z-scores must be calculated on all connected nodes in the
    # sex-age group, with mortality=0 for missing, so "above average" is relative
    # to the full diagnosis set.
    merged["z_betweenness"] = compute_z_score(merged["betweenness"])
    merged["z_mortality"] = compute_z_score(merged["mortality"])

    # Compute product of Z-scores
    merged["z_product"] = merged["z_betweenness"] * merged["z_mortality"]

    # Filter to nodes with BOTH z_betweenness > 0 AND z_mortality > 0
    # ("Only considering positive z-scores" = above average in both dimensions).
    merged_positive = merged[(merged["z_betweenness"] > 0) & (merged["z_mortality"] > 0)].copy()

    if len(merged_positive) == 0:
        logger.warning(f"No nodes with both positive z-scores for {sex} {age_group}")
        return pd.DataFrame()

    # Geometric mean for ranking/reporting (not for selection).
    merged_positive["z_geom_mean"] = np.sqrt(
        merged_positive["z_betweenness"] * merged_positive["z_mortality"]
    )

    # Select top X% by percentile threshold (manuscript: "top X% of z-score products").
    # Use (100 - top_percent)% percentile as cutoff so we keep nodes in the top X%
    # of the distribution (matches reference: quantile(threshold_percentile/100)).
    threshold = merged_positive["z_product"].quantile(1.0 - top_percentile)
    high_risk = merged_positive[merged_positive["z_product"] >= threshold].copy()
    high_risk = high_risk.sort_values("z_product", ascending=False)

    return high_risk


def identify_high_risk_edges(
    sex: str,
    age_group: str,
    mortality_df: pd.DataFrame,
    top_percentile: float = 0.05,
    min_mortality_diff: float = 0.30,
) -> pd.DataFrame:
    """
    Identify high-risk edges (high edge betweenness and high mortality difference).

    This implements the "high-mortality bridges" analysis from the paper:
    - Edges with high edge betweenness centrality (central in network structure)
    - AND high difference in mortality between connected nodes
    - Uses product of z-scores: z(edge_betweenness) × z(mortality_difference)
    - Selects top percentile of edges by z-product
    - Requires minimum absolute mortality difference (default: 30%)

    Note: The paper mentions bridges "peaking in older adults (female up to 89; male up to 68)",
    but our implementation may yield different counts due to:
    - Different data or thresholds
    - Different selection criteria (percentile vs. threshold)
    - Different minimum mortality difference requirements

    Args:
        sex: "Female" or "Male"
        age_group: Age group string (e.g., "0-9")
        mortality_df: DataFrame with columns: icd_code, mortality
        top_percentile: Top percentile to select (default: 0.05 for top 5%)
        min_mortality_diff: Minimum absolute mortality difference (default: 0.30)

    Returns:
        DataFrame with high-risk edges
    """
    # Load graph
    age_num = AGE_GROUP_MAP[age_group]
    adj_path = get_adjacency_matrix_path(sex, age_num)

    if not adj_path.exists():
        logger.warning(f"Adjacency matrix not found: {adj_path}")
        return pd.DataFrame()

    G = NetworkAnalyzer.load_adjacency_matrix(adj_path)

    # Create analyzer instance to reuse calculations
    analyzer = NetworkAnalyzer(G)

    # Get node-to-ICD mapping
    node_mapping = get_node_icd_mapping(sex, age_num)

    # Compute edge betweenness on FULL graph (normalized for individual edge analysis)
    # Reuse NetworkAnalyzer method which calculates on full graph and caches results
    edge_betweenness = analyzer.get_edge_betweenness(G, normalized=True, use_cache=True)

    # Filter to only connected nodes for edge processing
    G_filtered = analyzer.get_filtered_graph(G)

    # Filter mortality data for this stratum
    if "sex" in mortality_df.columns and "age_group" in mortality_df.columns:
        mortality_stratum = mortality_df[
            (mortality_df["sex"] == sex) & (mortality_df["age_group"] == age_group)
        ].copy()
    else:
        mortality_stratum = mortality_df.copy()

    # Create mortality lookup
    mortality_dict = dict(zip(mortality_stratum["icd_code"], mortality_stratum["mortality"]))

    # Process edges
    # Only include edges between connected nodes (nodes with at least one neighbor)
    filtered_nodes = set(G_filtered.nodes())
    edge_data = []
    for (u, v), eb in edge_betweenness.items():
        # Only process edges where both nodes are in the filtered graph
        if u not in filtered_nodes or v not in filtered_nodes:
            continue

        u_code = node_mapping.get(u, str(u))
        v_code = node_mapping.get(v, str(v))

        mortality_u = mortality_dict.get(u_code, None)
        mortality_v = mortality_dict.get(v_code, None)

        if mortality_u is not None and mortality_v is not None:
            mortality_diff = abs(mortality_u - mortality_v)
            if mortality_diff >= min_mortality_diff:
                edge_data.append(
                    {
                        "node1": u_code,
                        "node2": v_code,
                        "edge_betweenness": eb,
                        "mortality1": mortality_u,
                        "mortality2": mortality_v,
                        "mortality_diff": mortality_diff,
                    }
                )

    if not edge_data:
        return pd.DataFrame()

    edges_df = pd.DataFrame(edge_data)

    # Compute Z-scores
    edges_df["z_betweenness"] = compute_z_score(edges_df["edge_betweenness"])
    edges_df["z_mortality_diff"] = compute_z_score(edges_df["mortality_diff"])

    # Compute product of Z-scores
    edges_df["z_product"] = edges_df["z_betweenness"] * edges_df["z_mortality_diff"]

    # Select top percentile
    threshold = edges_df["z_product"].quantile(1 - top_percentile)
    high_risk_edges = edges_df[edges_df["z_product"] >= threshold].copy()
    high_risk_edges = high_risk_edges.sort_values("z_product", ascending=False)

    return high_risk_edges


def analyze_all_prevalence_degree(
    prevalence_path: Optional[Path] = None, year: int = 2014
) -> pd.DataFrame:
    """
    Analyze prevalence-degree correlation for all sex-age combinations.

    Args:
        prevalence_path: Path to prevalence data CSV
        year: Year to use for prevalence (default: 2014)

    Returns:
        DataFrame with results for all combinations
    """
    prevalence_df = load_prevalence_data(prevalence_path)
    all_results = []

    for sex in SEXES:
        for age_group in AGE_GROUPS.values():
            logger.info(f"Analyzing prevalence-degree correlation: {sex} {age_group}...")
            try:
                result = analyze_prevalence_degree_correlation(sex, age_group, prevalence_df, year)
                if not result.empty:
                    result["sex"] = sex
                    result["age_group"] = age_group
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {sex} {age_group}: {e}")
                continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def analyze_all_high_risk_nodes(
    mortality_df: pd.DataFrame, top_percentile: float = 0.20
) -> pd.DataFrame:
    """
    Identify high-risk nodes for all sex-age combinations.

    Args:
        mortality_df: DataFrame with mortality data (columns: icd_code, mortality, sex, age_group)
        top_percentile: Top percentile to select (default: 0.20)

    Returns:
        DataFrame with high-risk nodes for all combinations
    """
    all_results = []

    for sex in SEXES:
        for age_group in AGE_GROUPS.values():
            logger.info(f"Identifying high-risk nodes: {sex} {age_group}...")
            try:
                result = identify_high_risk_nodes(sex, age_group, mortality_df, top_percentile)
                if not result.empty:
                    result["sex"] = sex
                    result["age_group"] = age_group
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {sex} {age_group}: {e}")
                continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def analyze_all_high_risk_edges(
    mortality_df: pd.DataFrame,
    top_percentile: float = 0.05,
    min_mortality_diff: float = 0.30,
) -> pd.DataFrame:
    """
    Identify high-risk edges for all sex-age combinations.

    Args:
        mortality_df: DataFrame with mortality data (columns: icd_code, mortality, sex, age_group)
        top_percentile: Top percentile to select (default: 0.05)
        min_mortality_diff: Minimum absolute mortality difference (default: 0.30)

    Returns:
        DataFrame with high-risk edges for all combinations
    """
    all_results = []

    for sex in SEXES:
        for age_group in AGE_GROUPS.values():
            logger.info(f"Identifying high-risk edges: {sex} {age_group}...")
            try:
                result = identify_high_risk_edges(
                    sex, age_group, mortality_df, top_percentile, min_mortality_diff
                )
                if not result.empty:
                    result["sex"] = sex
                    result["age_group"] = age_group
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {sex} {age_group}: {e}")
                continue

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def load_mortality_data(
    mortality_dir: Optional[Path] = None, use_most_recent: bool = True
) -> pd.DataFrame:
    """
    Load mortality data from CSV files in interim/mortality directory.

    Args:
        mortality_dir: Directory containing mortality files. If None, uses default.
        use_most_recent: If True, uses most recent year (2014) data only

    Returns:
        DataFrame with mortality data (columns: icd_code, mortality, sex, age_group)
    """
    if mortality_dir is None:
        mortality_dir = INTERIM_DATA_DIR / "mortality"

    if not mortality_dir.exists():
        raise FileNotFoundError(f"Mortality directory not found: {mortality_dir}")

    # Load male and female mortality files
    male_path = mortality_dir / "mortality_diag_male.csv"
    female_path = mortality_dir / "mortality_diag_female.csv"

    if not male_path.exists() or not female_path.exists():
        raise FileNotFoundError(
            f"Mortality files not found. Expected: {male_path} and {female_path}"
        )

    # Load and combine
    male_df = pd.read_csv(male_path)
    female_df = pd.read_csv(female_path)

    # Add sex column
    male_df["sex"] = "Male"
    female_df["sex"] = "Female"

    # Combine
    combined_df = pd.concat([male_df, female_df], ignore_index=True)

    # Map age_10 to age_group string
    combined_df["age_group"] = combined_df["age_10"].map(AGE_10_TO_GROUP)

    # Filter to only age groups we use (1-8, which map to 0-9 through 70-79)
    combined_df = combined_df[combined_df["age_10"].isin(range(1, 9))]

    # Handle NaN mortality values (convert to 0)
    combined_df["mortality"] = pd.to_numeric(combined_df["mortality"], errors="coerce").fillna(0.0)

    # Select and rename columns
    result_df = combined_df[["icd_code", "mortality", "sex", "age_group"]].copy()

    logger.info(f"Loaded mortality data: {len(result_df)} rows")
    logger.info(f"  - Male: {len(result_df[result_df['sex'] == 'Male'])} rows")
    logger.info(f"  - Female: {len(result_df[result_df['sex'] == 'Female'])} rows")
    logger.info(f"  - Age groups: {sorted(result_df['age_group'].unique())}")

    return result_df


def create_mortality_template(output_path: Path) -> None:
    """
    Create a template CSV file for mortality data.

    Args:
        output_path: Path where template should be saved
    """
    template_data = {
        "icd_code": ["A00", "A01", "A02"],
        "mortality": [0.05, 0.10, 0.15],
        "sex": ["Female", "Female", "Female"],
        "age_group": ["0-9", "0-9", "0-9"],
    }
    template_df = pd.DataFrame(template_data)
    template_df.to_csv(output_path, index=False)
    logger.info(f"Created mortality data template at {output_path}")


def get_icd_code_mapping() -> Dict[str, str]:
    """
    Load ICD code to description mapping from DiagAll_Eng.csv.

    Returns:
        Dictionary mapping ICD codes to short descriptions
    """
    mapping_path = INTERIM_DATA_DIR / "DiagAll_Eng.csv"
    if not mapping_path.exists():
        logger.warning(f"ICD mapping file not found: {mapping_path}")
        return {}

    try:
        df = pd.read_csv(mapping_path)
        # Remove quotes from Code column if present
        df["Code"] = df["Code"].astype(str).str.strip('"')
        mapping = dict(zip(df["Code"], df["ShortDescription"]))
        logger.info(f"Loaded ICD code mapping: {len(mapping)} codes")
        return mapping
    except Exception as e:
        logger.warning(f"Error loading ICD mapping: {e}")
        return {}


def create_table2_format(
    prevalence_degree_df: pd.DataFrame,
    high_risk_nodes_df: pd.DataFrame,
    high_risk_edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Table 2 format showing counts of critical disease nodes and edges.

    Args:
        prevalence_degree_df: DataFrame from prevalence-degree analysis
        high_risk_nodes_df: DataFrame from high-risk nodes analysis
        high_risk_edges_df: DataFrame from high-risk edges analysis

    Returns:
        Formatted DataFrame matching Table 2 structure
    """
    age_groups = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]
    metrics = [
        "High degree outliers (high)",
        "High degree outliers (low)",
        "High-mortality sinks",
        "High-mortality bridges",
    ]

    # Create column structure: Metric, then Female/Male with age groups
    columns = ["Metric"]
    for sex in SEXES:
        for age_group in age_groups:
            columns.append(f"{sex}_{age_group}")

    table_rows = []

    for metric in metrics:
        row: dict = {"Metric": metric}

        for sex in SEXES:
            for age_group in age_groups:
                if metric == "High degree outliers (high)":
                    # Count nodes in high quintile (above 80th percentile)
                    # These are nodes with high degree relative to prevalence
                    if not prevalence_degree_df.empty:
                        count = len(
                            prevalence_degree_df[
                                (prevalence_degree_df["sex"] == sex)
                                & (prevalence_degree_df["age_group"] == age_group)
                                & (prevalence_degree_df["quintile_category"] == "high")
                            ]
                        )
                    else:
                        count = 0
                elif metric == "High degree outliers (low)":
                    # Count nodes in low quintile (below 20th percentile)
                    # These are nodes with low degree relative to prevalence
                    if not prevalence_degree_df.empty:
                        count = len(
                            prevalence_degree_df[
                                (prevalence_degree_df["sex"] == sex)
                                & (prevalence_degree_df["age_group"] == age_group)
                                & (prevalence_degree_df["quintile_category"] == "low")
                            ]
                        )
                    else:
                        count = 0
                elif metric == "High-mortality sinks":
                    # Count high-risk nodes (top 20%)
                    if not high_risk_nodes_df.empty:
                        count = len(
                            high_risk_nodes_df[
                                (high_risk_nodes_df["sex"] == sex)
                                & (high_risk_nodes_df["age_group"] == age_group)
                            ]
                        )
                    else:
                        count = 0
                elif metric == "High-mortality bridges":
                    # Count high-risk edges (top 5%)
                    if not high_risk_edges_df.empty:
                        count = len(
                            high_risk_edges_df[
                                (high_risk_edges_df["sex"] == sex)
                                & (high_risk_edges_df["age_group"] == age_group)
                            ]
                        )
                    else:
                        count = 0

                row[f"{sex}_{age_group}"] = count  # type: ignore

        table_rows.append(row)

    table2_df = pd.DataFrame(table_rows)
    return table2_df


def main() -> None:
    """Main function to run advanced analyses."""
    logger.info("Starting advanced network analyses...")

    # Analysis 1: Prevalence-Degree Correlation
    logger.info("\n" + "=" * 60)
    logger.info("Analysis 1: Prevalence-Degree Correlation")
    logger.info("=" * 60)

    prevalence_degree_df = pd.DataFrame()
    try:
        prevalence_degree_df = analyze_all_prevalence_degree()
        if not prevalence_degree_df.empty:
            output_path = PROCESSED_DATA_DIR / "prevalence_degree_analysis.csv"
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            prevalence_degree_df.to_csv(output_path, index=False)
            logger.success(f"Saved prevalence-degree analysis to {output_path}")

            # Summary statistics
            logger.info("\nPrevalence-Degree Correlation Summary:")
            summary = (
                prevalence_degree_df.groupby(["sex", "age_group", "quintile_category"])
                .size()
                .unstack(fill_value=0)
            )
            logger.info(f"\n{summary}")
        else:
            logger.warning("No prevalence-degree correlation results generated")
    except Exception as e:
        logger.error(f"Prevalence-degree analysis failed: {e}")

    # Analysis 2: High-Risk Nodes and Edges
    # Note: Requires mortality data
    logger.info("\n" + "=" * 60)
    logger.info("Analysis 2: High-Risk Disease Identification")
    logger.info("=" * 60)

    high_risk_nodes = pd.DataFrame()
    high_risk_edges = pd.DataFrame()

    # High-Risk Analysis (requires mortality data)
    try:
        mortality_df = load_mortality_data()

        # Analyze high-risk nodes
        logger.info("Identifying high-risk nodes...")
        high_risk_nodes = analyze_all_high_risk_nodes(mortality_df, top_percentile=0.20)
        if not high_risk_nodes.empty:
            output_path = PROCESSED_DATA_DIR / "high_risk_nodes.csv"
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            high_risk_nodes.to_csv(output_path, index=False)
            logger.success(f"Saved high-risk nodes to {output_path}")

        # Analyze high-risk edges
        logger.info("Identifying high-risk edges...")
        high_risk_edges = analyze_all_high_risk_edges(
            mortality_df, top_percentile=0.05, min_mortality_diff=0.30
        )
        if not high_risk_edges.empty:
            output_path = PROCESSED_DATA_DIR / "high_risk_edges.csv"
            high_risk_edges.to_csv(output_path, index=False)
            logger.success(f"Saved high-risk edges to {output_path}")
    except FileNotFoundError as e:
        logger.warning(f"Mortality data not found: {e}")
        logger.info("Skipping high-risk disease identification.")
    except Exception as e:
        logger.error(f"High-risk analysis failed: {e}")

    # Create Table 2
    logger.info("\n" + "=" * 60)
    logger.info("Creating Table 2: Critical Disease Nodes and Edges")
    logger.info("=" * 60)

    try:
        table2_df = create_table2_format(prevalence_degree_df, high_risk_nodes, high_risk_edges)
        if not table2_df.empty:
            output_path = PROCESSED_DATA_DIR / "table2_critical_diseases.csv"
            table2_df.to_csv(output_path, index=False)
            logger.success(f"Saved Table 2 to {output_path}")
            logger.info("\nTable 2 Format (Critical Disease Nodes and Edges):")
            logger.info(f"\n{table2_df.to_string(index=False)}")
        else:
            logger.warning("Could not create Table 2 - missing required data")
    except Exception as e:
        logger.error(f"Error creating Table 2: {e}")


if __name__ == "__main__":
    main()
