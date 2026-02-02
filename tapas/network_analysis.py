"""Network analysis module for calculating network properties from comorbidity networks."""

from pathlib import Path

import pandas as pd
from loguru import logger

from tapas.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from tapas.features import NetworkAnalyzer

# Age group mapping: age_1 = 0-9, age_2 = 10-19, ..., age_8 = 70-79
AGE_GROUPS = {
    1: "0-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70-79",
}

SEXES = ["Female", "Male"]


def get_adjacency_matrix_path(sex: str, age_num: int) -> Path:
    """
    Get the path to an adjacency matrix CSV file for a given sex and age group.

    Args:
        sex: "Female" or "Male"
        age_num: Age group number (1-8)

    Returns:
        Path to the adjacency matrix CSV file
    """
    adj_matrices_dir = INTERIM_DATA_DIR / "extracted" / "Data" / "3.AdjacencyMatrices"
    filename = f"Adj_Matrix_{sex}_ICD_age_{age_num}.csv"
    return adj_matrices_dir / filename


def analyze_all_networks() -> pd.DataFrame:
    """
    Analyze all networks and create a table matching Table 1 format.

    Returns:
        DataFrame with network properties for all sex/age combinations
    """
    results = []

    for sex in SEXES:
        for age_num, age_group in AGE_GROUPS.items():
            file_path = get_adjacency_matrix_path(sex, age_num)

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"Analyzing {sex} age group {age_group} (age_{age_num})...")

            try:
                # Load graph using NetworkAnalyzer
                G = NetworkAnalyzer.load_adjacency_matrix(file_path)
                # Calculate properties using NetworkAnalyzer
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
    """
    Create a table in the format of Table 1 from the paper.
    Table structure: Rows are metrics, columns are Female/Male with age groups as sub-columns.

    Args:
        df: DataFrame with network properties

    Returns:
        Formatted DataFrame matching Table 1 structure
    """
    metrics = [
        "Connected Nodes",
        "Degree",
        "Average Path",
        "Betweenness",
        "Closeness",
        "Modularity",
        "Clustering",
    ]
    age_groups = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

    # Create column structure: Metric, then Female/Male with age groups
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
                    row[f"{sex}_{age_group}"] = None  # type: ignore
                    continue

                value = None
                if metric == "Connected Nodes":
                    value = int(age_data.iloc[0]["connected_nodes"])
                elif metric == "Degree":
                    value = age_data.iloc[0]["degree"]
                elif metric == "Average Path":
                    value = age_data.iloc[0]["average_path"]
                elif metric == "Betweenness":
                    value = age_data.iloc[0]["betweenness"]
                elif metric == "Closeness":
                    value = age_data.iloc[0]["closeness"]
                elif metric == "Modularity":
                    value = age_data.iloc[0]["modularity"]
                elif metric == "Clustering":
                    value = age_data.iloc[0]["clustering"]

                row[f"{sex}_{age_group}"] = value  # type: ignore

        table_rows.append(row)

    table1_df = pd.DataFrame(table_rows)
    return table1_df


def main() -> None:
    """Main function to run network analysis and generate table."""
    logger.info("Starting network analysis...")

    # Analyze all networks
    df = analyze_all_networks()

    if df.empty:
        logger.error("No network data was analyzed. Check file paths.")
        return

    # Save detailed results to CSV
    output_path = PROCESSED_DATA_DIR / "network_properties_table.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved network properties table to {output_path}")

    # Display summary
    logger.info("\nNetwork Properties Summary:")
    logger.info(f"\n{df.to_string(index=False)}")

    # Create a formatted table matching Table 1 structure
    logger.info("\nCreating Table 1 format...")
    table1_df = create_table1_format(df)

    # Save formatted table
    table1_path = PROCESSED_DATA_DIR / "network_properties_table1_format.csv"
    table1_df.to_csv(table1_path, index=False)
    logger.success(f"Saved Table 1 format to {table1_path}")

    # Display Table 1 format
    logger.info("\nTable 1 Format (Network Properties):")
    logger.info(f"\n{table1_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
