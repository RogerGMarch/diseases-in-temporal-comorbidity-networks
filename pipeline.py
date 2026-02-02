"""
Pipeline script for downloading and processing comorbidity networks data.

This script orchestrates the entire data pipeline:
1. Download dataset from figshare
2. Extract and process the data
3. Analyze network properties and generate Table 1
"""

from pathlib import Path

from loguru import logger
import typer

from tapas.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from tapas.dataset import download_from_figshare, process_dataset
from tapas.network_analysis import analyze_all_networks, create_table1_format
from tapas.advanced_analysis import (
    analyze_all_prevalence_degree,
    analyze_all_high_risk_nodes,
    analyze_all_high_risk_edges,
    load_mortality_data,
)

app = typer.Typer()


@app.command()
def run(
    url: str = "https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553?file=52015403",
    download_path: Path = RAW_DATA_DIR / "comorbidity_networks_data.zip",
    extract_to: Path = INTERIM_DATA_DIR / "extracted",
    output_dir: Path = PROCESSED_DATA_DIR,
    skip_download: bool = False,
    run_advanced_analysis: bool = True,
) -> None:
    """
    Run the complete data pipeline: download, process, and analyze the dataset.

    Args:
        url: Figshare URL for the dataset
        download_path: Path where the downloaded file should be saved
        extract_to: Directory to extract the zip file to (interim)
        output_dir: Directory to save processed data
        skip_download: If True, skip download step (assumes file already exists)
        run_advanced_analysis: If True, run advanced analyses (prevalence-degree, high-risk)
    """
    logger.info("=" * 60)
    logger.info("Starting Comorbidity Networks Data Pipeline")
    logger.info("=" * 60)

    # Step 1: Download data
    if not skip_download:
        step_num = "1/4" if run_advanced_analysis else "1/3"
        logger.info(f"\n[Step {step_num}] Downloading dataset from figshare...")
        if download_path.exists():
            logger.warning(f"File already exists: {download_path}")
            logger.info("Skipping download, using existing file.")
            logger.info("To force re-download, delete the file or use --skip-download=False")
        else:
            try:
                download_from_figshare(url, download_path)
                logger.success("Download complete!")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise
    else:
        step_num = "1/4" if run_advanced_analysis else "1/3"
        logger.info(f"\n[Step {step_num}] Skipping download (skip_download=True)")
        if not download_path.exists():
            logger.error(f"Download file not found: {download_path}")
            logger.info("Set skip_download=False or download the file first.")
            raise FileNotFoundError(f"Download file not found: {download_path}")

    # Step 2: Process data
    step_num = "2/4" if run_advanced_analysis else "2/3"
    logger.info(f"\n[Step {step_num}] Processing dataset...")
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

    # Step 3: Network analysis
    step_num = "3/4" if run_advanced_analysis else "3/3"
    logger.info(f"\n[Step {step_num}] Analyzing network properties...")
    try:
        df = analyze_all_networks()
        if df.empty:
            logger.warning("No network data was analyzed. Check file paths.")
        else:
            # Save detailed results
            output_path = output_dir / "network_properties_table.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.success(f"Saved network properties table to {output_path}")

            # Create and save Table 1 format
            table1_df = create_table1_format(df)
            table1_path = output_dir / "network_properties_table1_format.csv"
            table1_df.to_csv(table1_path, index=False)
            logger.success(f"Saved Table 1 format to {table1_path}")
            logger.success("Network analysis complete!")
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise

    # Step 4: Advanced analysis (optional)
    if run_advanced_analysis:
        logger.info("\n[Step 4/4] Running advanced analyses...")
        try:
            # Prevalence-Degree Correlation Analysis
            logger.info("Analyzing prevalence-degree correlation...")
            prevalence_degree_df = analyze_all_prevalence_degree()
            if not prevalence_degree_df.empty:
                output_path = output_dir / "prevalence_degree_analysis.csv"
                prevalence_degree_df.to_csv(output_path, index=False)
                logger.success(f"Saved prevalence-degree analysis to {output_path}")

            # High-Risk Analysis (requires mortality data)
            try:
                logger.info("Identifying high-risk nodes and edges...")
                mortality_df = load_mortality_data()

                high_risk_nodes = analyze_all_high_risk_nodes(mortality_df, top_percentile=0.20)
                if not high_risk_nodes.empty:
                    output_path = output_dir / "high_risk_nodes.csv"
                    high_risk_nodes.to_csv(output_path, index=False)
                    logger.success(f"Saved high-risk nodes to {output_path}")

                high_risk_edges = analyze_all_high_risk_edges(
                    mortality_df, top_percentile=0.05, min_mortality_diff=0.30
                )
                if not high_risk_edges.empty:
                    output_path = output_dir / "high_risk_edges.csv"
                    high_risk_edges.to_csv(output_path, index=False)
                    logger.success(f"Saved high-risk edges to {output_path}")
            except FileNotFoundError as e:
                logger.warning(f"Mortality data not found: {e}")
                logger.info("Skipping high-risk analysis. Provide mortality data to enable this analysis.")
            except Exception as e:
                logger.warning(f"High-risk analysis failed: {e}")
                logger.info("Continuing with pipeline completion...")

            logger.success("Advanced analysis complete!")
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}")
            logger.info("Continuing with pipeline completion...")

    logger.info("\n" + "=" * 60)
    logger.success("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Raw data: {download_path}")
    logger.info(f"Extracted data: {extract_to}")
    logger.info(f"Processed data: {output_dir}")


if __name__ == "__main__":
    app()
