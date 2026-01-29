# Diseases in Temporal Comorbidity Networks

A research project for analyzing diseases in temporal comorbidity networks using population-wide health data.

## Overview

This project processes comorbidity network data from a large-scale health dataset (8.9M hospital patients, 1997-2014) to analyze disease relationships and temporal patterns.

## Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Setup

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Create virtual environment and install dependencies

```bash
# Create virtual environment
make create_environment

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .\.venv\Scripts\activate

# Install dependencies
uv sync
```

Or simply run:

```bash
make requirements
```

This will:
- Create a virtual environment (if it doesn't exist)
- Install all project dependencies from `pyproject.toml`

## Project Structure

```text
.
├── data/                    # Data directory (created after download)
│   ├── raw/                 # Raw downloaded data
│   ├── interim/             # Intermediate processing files
│   ├── processed/           # Final processed datasets
│   └── external/            # External data sources
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks for analysis
├── reports/                 # Generated reports and figures
│   └── figures/            # Visualization outputs
├── tapas/                   # Main package
│   ├── config.py           # Configuration and paths
│   ├── dataset.py          # Data download and processing
│   ├── features.py         # Feature engineering
│   ├── plots.py            # Plotting utilities
│   └── modeling/           # Model training and prediction
│       ├── train.py        # Training scripts
│       └── predict.py      # Prediction scripts
├── pipeline.py             # Main pipeline script
├── pyproject.toml          # Project dependencies
└── Makefile                # Make commands for common tasks
```

## Usage

### Download and Process Data

The easiest way to download and process the dataset:

```bash
# Using the pipeline script
python pipeline.py run

# Or using make
make data
```

This will:

1. Download the dataset from Figshare (~GB file)
2. Extract the zip file
3. Process prevalence data and adjacency matrices
4. Save processed data to `data/processed/`

#### Pipeline Options

```bash
# Skip download (if file already exists)
python pipeline.py run --skip-download

# Custom download path
python pipeline.py run --download-path data/raw/my_data.zip

# Custom extraction directory
python pipeline.py run --extract-to data/interim/custom_extract

# Custom output directory
python pipeline.py run --output-dir data/processed/custom_output
```

### Individual Steps

You can also run individual steps:

```bash
# Download only
python -m tapas.dataset download

# Process only (requires downloaded file)
python -m tapas.dataset main
```

### Training Models

```bash
python -m tapas.modeling.train
```

### Making Predictions

```bash
python -m tapas.modeling.predict
```

## Development

### Code Formatting and Linting

```bash
# Check formatting and linting
make lint

# Auto-fix formatting and linting issues
make format
```

### Clean Up

```bash
# Remove compiled Python files
make clean
```

## Data Source

The dataset is downloaded from:

- **Figshare**: [Comorbidity Networks From Population-Wide Health Data](https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553)

The dataset includes:
- Prevalence data (Sex, Age, Year, ICD codes)
- Contingency tables
- Adjacency matrices
- Graph files (GEXF format)

## Configuration

Project paths are configured in `tapas/config.py`. The configuration automatically:
- Detects the project root directory
- Creates necessary data directories
- Sets up paths for raw, interim, processed, and external data

## Dependencies

Main dependencies (see `pyproject.toml` for full list):
- `pandas` - Data manipulation
- `loguru` - Logging
- `typer` - CLI framework
- `requests` - HTTP requests for downloads
- `tqdm` - Progress bars
- `ruff` - Code formatting and linting

## Troubleshooting

### Download Issues

If the download fails:

1. Check your internet connection
2. Verify the Figshare URL is accessible
3. Try downloading manually and placing the file in `data/raw/comorbidity_networks_data.zip`

### Path Issues

If you encounter path-related errors:

- Ensure you're running commands from the project root directory
- Check that `tapas/config.py` correctly identifies the project root
- Verify all data directories are created (they should be created automatically)

### Virtual Environment

If you have issues with the virtual environment:

```bash
# Remove existing environment
rm -rf .venv

# Recreate and install
make create_environment
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
uv sync
```

## Make Commands

Run `make help` to see all available commands:

```bash
make help
```

Common commands:

- `make requirements` - Install dependencies
- `make data` - Download and process data
- `make lint` - Check code quality
- `make format` - Format code
- `make clean` - Clean compiled files
- `make create_environment` - Create virtual environment

## License

[Add your license information here]

## Citation

If you use this code or dataset, please cite the original data source:
[Add citation information here]
