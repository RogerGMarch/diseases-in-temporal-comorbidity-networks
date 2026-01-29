"""
Pipeline script for downloading and processing comorbidity networks data.

This script orchestrates the entire data pipeline:
1. Download dataset from figshare
2. Extract and process the data
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

app = typer.Typer()


@app.command()
def run(
    url: str = "https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553?file=52015403",
    download_path: Path = RAW_DATA_DIR / "comorbidity_networks_data.zip",
    extract_to: Path = INTERIM_DATA_DIR / "extracted",
    output_dir: Path = PROCESSED_DATA_DIR,
    skip_download: bool = False,
) -> None:
    """
    Run the complete data pipeline: download and process the dataset.

    Args:
        url: Figshare URL for the dataset
        download_path: Path where the downloaded file should be saved
        extract_to: Directory to extract the zip file to (interim)
        output_dir: Directory to save processed data
        skip_download: If True, skip download step (assumes file already exists)
    """
    logger.info("=" * 60)
    logger.info("Starting Comorbidity Networks Data Pipeline")
    logger.info("=" * 60)

    # Step 1: Download data
    if not skip_download:
        logger.info("\n[Step 1/2] Downloading dataset from figshare...")
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
        logger.info("\n[Step 1/2] Skipping download (skip_download=True)")
        if not download_path.exists():
            logger.error(f"Download file not found: {download_path}")
            logger.info("Set skip_download=False or download the file first.")
            raise FileNotFoundError(f"Download file not found: {download_path}")

    # Step 2: Process data
    logger.info("\n[Step 2/2] Processing dataset...")
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

    logger.info("\n" + "=" * 60)
    logger.success("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Raw data: {download_path}")
    logger.info(f"Extracted data: {extract_to}")
    logger.info(f"Processed data: {output_dir}")


if __name__ == "__main__":
    app()
