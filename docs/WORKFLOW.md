# Analysis Workflow

This document outlines the workflow for reproducing the paper's network and mortality analyses, from data processing to generating final tables and figures.

## Setup & Prerequisites

Before running the analysis, ensure your environment is configured correctly:

1. **Python Version:** Ensure Python 3.10.13 is installed (as specified in `.python-version`).
2. **Data Availability:** Place the downloaded dataset from Figshare (and required mortality CSVs) into the `data/raw/` and `data/interim/mortality/` directories.
3. **Environment Setup:** Install dependencies and sync the environment using `uv`.
   ```bash
   uv sync


## Running the Pipeline
Execute the following commands in order to generate all required tables, data, and plots.


#### 1. Generate robustness summary table
```uv run analysis/summary.py```

#### 2. Identify network outliers (Generates Table S2)
```uv run analysis/outliers.py```

#### 3. Analyze high-mortality sinks (Generates mortality sinks CSV)
```uv run analysis/mortality.py```

#### 4. Find the intersection of critical nodes
```uv run analysis/intersection.py```

#### 5. Identify critical bridge edges (Generates bridges-edges.csv)
```uv run analysis/bridges.py```

#### 6. Generate visualizations and plots for outliers
```uv run analysis/panel_outliers.py```

#### 7. Generate final robustness data
```uv run analysis/robustness```

## References

- **Data Source**: [Comorbidity Networks From Population-Wide Health Data](https://figshare.com/articles/dataset/27102553)
- **Paper**: [Add citation to your paper here]
- **Code Repository**: [Add GitHub/GitLab URL here]
