# diseases in temporal comorbidity networks

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project for analyzing diseases in temporal comorbidity networks using population-wide health data.

## Getting Started

### Installation

Install the project dependencies using `uv`:

```bash
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

### Data Pipeline

The project includes a pipeline script to download and process the comorbidity networks dataset from figshare.

#### Quick Start

Run the complete pipeline (download + process):

```bash
uv run tapas.pipeline
```

This will:
1. Download the dataset from figshare (~203MB) to `data/raw/comorbidity_networks_data.zip`
2. Extract the data to `data/interim/extracted/`
3. Process the data and save to `data/processed/`

#### Pipeline Options

```bash
# Skip download if file already exists
uv run tapas.pipeline --skip-download

# Custom download path
uv run tapas.pipeline --download-path /path/to/data.zip

# Custom output directory
uv run tapas.pipeline --output-dir /path/to/output

# Custom extraction directory
uv run tapas.pipeline --extract-to /path/to/extract
```

#### Individual Commands

You can also run individual steps separately:

```bash
# Download only
uv run tapas.dataset download

# Process only (requires downloaded file)
uv run tapas.dataset main
```

### Dataset

The dataset contains:
- **Prevalence data**: Disease prevalence by sex, age group, year, and ICD code
- **Adjacency matrices**: Comorbidity network adjacency matrices (84 files)
- **Contingency tables**: Disease co-occurrence contingency tables
- **Graph files**: Network graphs in GEXF format

Processed outputs:
- `data/processed/prevalence_data.csv` - Processed prevalence data
- `data/processed/adjacency_matrices_metadata.csv` - Metadata about adjacency matrices
- `data/interim/extracted/Data/` - Full extracted dataset structure

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tapas and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tapas   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tapas a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── pipeline.py             <- Main pipeline script for downloading and processing data
    │
    └── plots.py                <- Code to create visualizations
```

## Data Source

The dataset is downloaded from:
- **Figshare**: [Comorbidity Networks From Population-Wide Health Data: Aggregated Data of 8.9M Hospital Patients (1997-2014)](https://figshare.com/articles/dataset/Comorbidity_Networks_From_Population-Wide_Health_Data_Aggregated_Data_of_8_9M_Hospital_Patients_1997-2014_/27102553)

--------

