"""
Pipeline script for downloading and processing comorbidity networks data.

This script orchestrates the entire data pipeline:
1. Download dataset from figshare
2. Extract and process the data
3. Analyze network properties and generate Table 1
4. Run advanced analyses

Usage:
    python pipeline.py              # Run everything
    python pipeline.py --skip-download  # Skip download if file exists
"""

from pathlib import Path

from loguru import logger

from tapas.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from tapas.config import (
    AGE_GROUPS,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEXES,
)
from tapas.dataset import download_from_figshare, process_dataset
from tapas.features import NetworkAnalyzer
from tapas.analysis.outliers import get_all_outliers_exact
from tapas.analysis.mortality import main as run_mortality_analysis
from tapas.analysis.bridges import main as run_bridges_analysis
from tapas.visualization import generate_all_plots


def main() -> None:
    """Run the complete data pipeline."""
    # Configuration
    url = "https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553?file=52015403"
    download_path = RAW_DATA_DIR / "comorbidity_networks_data.zip"
    extract_to = INTERIM_DATA_DIR / "extracted"
    output_dir = PROCESSED_DATA_DIR
    logger.info("=" * 60)
    logger.info("Starting Comorbidity Networks Data Pipeline")
    logger.info("=" * 60)

    # Step 1: Download data
    logger.info("\n[Step 1/4] Downloading dataset from figshare...")
    if download_path.exists():
        logger.warning(f"File already exists: {download_path}")
        logger.info("Using existing file. Delete it to force re-download.")
    else:
        try:
            download_from_figshare(url, download_path)
            logger.success("Download complete!")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    # Step 2: Process data
    logger.info("\n[Step 2/4] Processing dataset...")
    try:
        process_dataset(
            input_path=download_path,
            output_dir=output_dir,
            extract_to=extract_to,
        )
        logger.success("Processing complete!")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

    # Step 3: Network analysis (properties table)
    logger.info("\n[Step 3/5] Analyzing network properties...")
    try:
        import pandas as pd
        rows = []
        for sex in SEXES:
            for age_id, age_range in AGE_GROUPS.items():
                paths = NetworkAnalyzer._resolve_paths(sex, age_id)
                adj_path = paths["adjacency"]
                if not adj_path.exists():
                    continue
                try:
                    G = NetworkAnalyzer.load_adjacency_matrix(adj_path)
                    analyzer = NetworkAnalyzer(G)
                    props = analyzer.calculate_all_properties(G)
                    props.update({"Sex": sex, "Age_Group": age_id, "Age_Range": age_range})
                    rows.append(props)
                except Exception as e:
                    logger.warning(f"Skipping {sex} age {age_id}: {e}")

        if rows:
            df_props = pd.DataFrame(rows)
            output_path = output_dir / "network_properties_table.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            df_props.to_csv(output_path, index=False)
            logger.success(f"Saved network properties table to {output_path}")
        else:
            logger.warning("No network data was analyzed. Check file paths.")
        logger.success("Network analysis complete!")
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise

    # Step 4: Advanced analyses
    logger.info("\n[Step 4/5] Running advanced analyses...")
    try:
        # 4a. Outlier / prevalence-degree analysis
        logger.info("Detecting degree-prevalence outliers...")
        df_outliers = get_all_outliers_exact()
        if not df_outliers.empty:
            df_outliers.to_csv(output_dir / "Outliers_EXACT.csv", index=False)
            logger.success("Saved Outliers_EXACT.csv")

        # 4b. High-mortality sinks
        logger.info("Identifying high-mortality sinks...")
        try:
            run_mortality_analysis()
        except Exception as e:
            logger.warning(f"Mortality analysis failed: {e}")

        # 4c. Bridge edges
        logger.info("Identifying bridge edges...")
        try:
            run_bridges_analysis()
        except Exception as e:
            logger.warning(f"Bridge analysis failed: {e}")

        logger.success("Advanced analysis complete!")
    except Exception as e:
        logger.warning(f"Advanced analysis failed: {e}")
        logger.info("Continuing with pipeline completion...")

    # Step 5: Generate visualizations
    logger.info("\n[Step 5/5] Generating plots...")
    try:
        generate_all_plots()
        logger.success("Plots saved to reports/figures/")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")
        logger.info("Continuing with pipeline completion...")

    logger.info("\n" + "=" * 60)
    logger.success("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Raw data: {download_path}")
    logger.info(f"Extracted data: {extract_to}")
    logger.info(f"Processed data: {output_dir}")
    logger.info(f"Plots: reports/figures/")


if __name__ == "__main__":
    main()
