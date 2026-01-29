import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm
import typer

from tapas.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


def download_from_figshare(
    url: str,
    output_path: Path,
    chunk_size: int = 8192,
) -> None:
    """
    Download a file from figshare.

    Args:
        url: Figshare URL containing file ID (e.g., .../27102553?file=52015403)
        output_path: Path where the file should be saved
        chunk_size: Size of chunks for downloading (default: 8192 bytes)
    """
    # Extract file ID from URL
    # Pattern: .../article_id?file=file_id
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    if "file" not in query_params:
        # Try to extract from URL path if not in query params
        # Pattern: .../article_id/files/file_id
        match = re.search(r"/files/(\d+)", url)
        if match:
            file_id = match.group(1)
        else:
            raise ValueError(f"Could not extract file ID from URL: {url}")
    else:
        file_id = query_params["file"][0]
    
    logger.info(f"Downloading file ID {file_id} from figshare...")
    
    # Construct direct download URL
    download_url = f"https://ndownloader.figshare.com/files/{file_id}"
    
    # Make request with stream=True to download in chunks
    response = requests.get(download_url, stream=True, timeout=30)
    response.raise_for_status()
    
    # Get total file size from headers
    total_size = int(response.headers.get("content-length", 0))
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    logger.success(f"Downloaded {output_path.name} to {output_path.parent}")


@app.command()
def download(
    url: str = "https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553?file=52015403",
    output_path: Path = RAW_DATA_DIR / "comorbidity_networks_data.zip",
) -> None:
    """
    Download the main dataset from figshare.
    
    Args:
        url: Figshare URL for the dataset
        output_path: Path where the downloaded file should be saved
    """
    download_from_figshare(url, output_path)


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extract zip file to a directory, skipping __MACOSX and .DS_Store files.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to

    Returns:
        Path to the extracted data directory
    """
    logger.info(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of files, excluding __MACOSX and .DS_Store
        file_list = [
            f
            for f in zip_ref.namelist()
            if not f.startswith("__MACOSX") and not f.endswith(".DS_Store")
        ]

        # Extract files with progress bar
        for file in tqdm(file_list, desc="Extracting files"):
            zip_ref.extract(file, extract_to)

    # Find the Data directory
    data_dir = extract_to / "Data"
    if data_dir.exists():
        logger.success(f"Extracted data to {data_dir}")
        return data_dir
    else:
        logger.warning(f"Data directory not found in {extract_to}")
        return extract_to


def process_dataset(
    input_path: Path,
    output_dir: Path,
    extract_to: Path,
) -> None:
    """
    Process the comorbidity networks dataset.

    Extracts the zip file, processes CSV files (prevalence and adjacency matrices),
    and saves processed data to the processed directory.

    Args:
        input_path: Path to the downloaded zip file
        output_dir: Directory to save processed data
        extract_to: Directory to extract the zip file to (interim)
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run 'python -m tapas.dataset download' first to download the dataset.")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Starting dataset processing...")

    # Step 1: Extract zip file
    data_dir = extract_zip(input_path, extract_to)

    # Step 2: Process prevalence data
    prevalence_path = data_dir / "1.Prevalence" / "Prevalence_Sex_Age_Year_ICD.csv"
    if prevalence_path.exists():
        logger.info("Processing prevalence data...")
        try:
            prevalence_df = pd.read_csv(prevalence_path)
            logger.info(f"Loaded prevalence data: {len(prevalence_df)} rows, {len(prevalence_df.columns)} columns")
            
            # Save processed prevalence data
            output_dir.mkdir(parents=True, exist_ok=True)
            prevalence_output = output_dir / "prevalence_data.csv"
            prevalence_df.to_csv(prevalence_output, index=False)
            logger.success(f"Saved processed prevalence data to {prevalence_output}")
        except Exception as e:
            logger.error(f"Error processing prevalence data: {e}")

    # Step 3: Process adjacency matrices
    adj_matrices_dir = data_dir / "3.AdjacencyMatrices"
    if adj_matrices_dir.exists():
        logger.info("Processing adjacency matrices...")
        adj_files = list(adj_matrices_dir.glob("*.csv"))
        
        if adj_files:
            logger.info(f"Found {len(adj_files)} adjacency matrix files")
            
            # Process a sample of adjacency matrices (you can modify this logic)
            processed_matrices = []
            for adj_file in tqdm(adj_files[:5], desc="Processing matrices"):  # Process first 5 as example
                try:
                    adj_df = pd.read_csv(adj_file)
                    # Store metadata about the matrix
                    processed_matrices.append({
                        "filename": adj_file.name,
                        "shape": adj_df.shape,
                        "file_path": str(adj_file.relative_to(data_dir)),
                    })
                except Exception as e:
                    logger.warning(f"Error processing {adj_file.name}: {e}")
            
            # Save metadata about adjacency matrices
            if processed_matrices:
                matrices_metadata = pd.DataFrame(processed_matrices)
                metadata_output = output_dir / "adjacency_matrices_metadata.csv"
                matrices_metadata.to_csv(metadata_output, index=False)
                logger.success(f"Saved adjacency matrices metadata to {metadata_output}")
        else:
            logger.warning("No adjacency matrix CSV files found")

    # Step 4: List available data files
    logger.info("Dataset structure:")
    logger.info(f"  - Prevalence data: {prevalence_path.exists()}")
    logger.info(f"  - Adjacency matrices: {adj_matrices_dir.exists()}")
    
    contingency_dir = data_dir / "2.ContingencyTables"
    graphs_dir = data_dir / "4.Graphs-gexffiles"
    logger.info(f"  - Contingency tables: {contingency_dir.exists()}")
    logger.info(f"  - Graph files: {graphs_dir.exists()}")

    logger.success("Dataset processing complete!")
    logger.info(f"Processed data saved to: {output_dir}")
    logger.info(f"Extracted data available at: {data_dir}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "comorbidity_networks_data.zip",
    output_dir: Path = PROCESSED_DATA_DIR,
    extract_to: Path = INTERIM_DATA_DIR / "extracted",
) -> None:
    """
    Process the comorbidity networks dataset (CLI command).

    Extracts the zip file, processes CSV files (prevalence and adjacency matrices),
    and saves processed data to the processed directory.

    Args:
        input_path: Path to the downloaded zip file
        output_dir: Directory to save processed data
        extract_to: Directory to extract the zip file to (interim)
    """
    process_dataset(input_path, output_dir, extract_to)


if __name__ == "__main__":
    app()
