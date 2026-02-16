"""
Network property calculations for comorbidity networks.

This analysis generates Table 1 of the paper by computing network properties
across all 16 demographic-stratified networks (Female/Male × 8 age groups).

Computed Metrics:
- Connected Nodes: Number of diseases with at least one comorbidity connection
- Average Degree: Mean number of comorbidities per disease
- Average Path Length: Mean shortest path distance between diseases
- Betweenness Centrality: Identifies "bridge" diseases in network
- Closeness Centrality: Measures disease accessibility within network
- Modularity: Quantifies strength of community structure
- Clustering Coefficient: Tendency for diseases to form tightly connected groups

Methodology:
- All metrics computed on full networks before filtering
- Isolated nodes excluded from centrality calculations
- Normalization applied for cross-network comparability
- Community detection via Louvain algorithm

Paper Reference:
- Table 1: Network properties stratified by sex and age group
- Methods section: Network metric definitions and calculation procedures
- Results section: Interpretation of network properties

Output Files:
- network_properties_table.csv: Complete results with all metrics
- network_properties_table1_format.csv: Formatted for publication (Table 1)
"""

from pathlib import Path
from loguru import logger
import pandas as pd

from tapas.config import PROCESSED_DATA_DIR, SEXES, AGE_GROUPS, INTERIM_DATA_DIR, DATA_DIR
from tapas.features import NetworkAnalyzer


def get_adjacency_matrix_path(sex: str, age_num: int) -> Path:
    """
    Get the path to the adjacency matrix file for a given sex and age group.
    
    Args:
        sex: "Female" or "Male"
        age_num: Age group number (1-8)
        
    Returns:
        Path to the adjacency matrix CSV file
    """
    # Use the same logic as NetworkAnalyzer._resolve_paths
    base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
    if not base_data_path.exists():
        base_data_path = DATA_DIR
    
    adj_filename = f"Adj_Matrix_{sex}_ICD_age_{age_num}.csv"
    return base_data_path / "3.AdjacencyMatrices" / adj_filename


def analyze_all_networks() -> pd.DataFrame:
    """Analyze all networks and create a table matching Table 1 format."""
    results = []

    for sex in SEXES:
        for age_num, age_group in AGE_GROUPS.items():
            # Use new centralized path resolution via private helper or rebuild logic
            # Since _resolve_paths is internal, we use the load_adjacency_matrix and path resolution logic
            # exposed via NetworkAnalyzer or reconstruct the path here using features.py logic
            
            # Reusing the private helper from features for consistency if possible, 
            # or strictly speaking, we can just instantiate the paths. 
            # Let's use the features.py class to get data.
            
            paths = NetworkAnalyzer._resolve_paths(sex, age_num)
            file_path = paths["adjacency"]

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"Analyzing {sex} age group {age_group} (age_{age_num})...")

            try:
                G = NetworkAnalyzer.load_adjacency_matrix(file_path)
                analyzer = NetworkAnalyzer(G)
                properties = analyzer.calculate_all_properties(filter_isolated=True)

                results.append(
                    {
                        "sex": sex,
                        "age_group": age_group,
                        "age_num": age_num,
                        "connected_nodes": properties["connected_nodes"],
                        "degree": round(properties["degree"], 2),
                        "average_path": round(properties["average_path"], 2),
                        "betweenness": round(properties["betweenness"], 2),
                        "closeness": round(properties["closeness"], 2),
                        "modularity": round(properties["modularity"], 2),
                        "clustering": round(properties["clustering"], 2),
                    }
                )
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                continue

    df = pd.DataFrame(results)
    return df

def create_table1_format(df: pd.DataFrame) -> pd.DataFrame:
    """Create a table in the format of Table 1 from the paper."""
    metrics = ["Connected Nodes", "Degree", "Average Path", "Betweenness", "Closeness", "Modularity","Clustering"]
    age_groups = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

    columns = ["Metric"]
    for sex in SEXES:
        for age_group in age_groups:
            columns.append(f"{sex}_{age_group}")

    table_rows = []
    for metric in metrics:
        row: dict = {"Metric": metric}
        for sex in SEXES:
            sex_data = df[df["sex"] == sex].sort_values("age_num")
            for age_group in age_groups:
                age_data = sex_data[sex_data["age_group"] == age_group]
                if age_data.empty:
                    row[f"{sex}_{age_group}"] = None
                    continue
                
                key_map = {
                    "Connected Nodes": "connected_nodes", "Degree": "degree", 
                    "Average Path": "average_path", "Betweenness": "betweenness", 
                    "Closeness": "closeness", "Modularity": "modularity", 
                    "Clustering": "clustering"
                }
                val_key = key_map.get(metric)
                value = age_data.iloc[0][val_key]
                if metric == "Connected Nodes": value = int(value)
                row[f"{sex}_{age_group}"] = value
        table_rows.append(row)

    return pd.DataFrame(table_rows)

def main() -> None:
    """Main function to run network analysis and generate table."""
    logger.info("Starting network analysis...")
    df = analyze_all_networks()
    if df.empty:
        logger.error("No network data was analyzed.")
        return

    output_path = PROCESSED_DATA_DIR / "network_properties_table.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved network properties table to {output_path}")

    logger.info("\nCreating Table 1 format...")
    table1_df = create_table1_format(df)
    table1_path = PROCESSED_DATA_DIR / "network_properties_table1_format.csv"
    table1_df.to_csv(table1_path, index=False)
    logger.success(f"Saved Table 1 format to {table1_path}")
    logger.info(f"\n{table1_df.to_string(index=False)}")

if __name__ == "__main__":
    main()