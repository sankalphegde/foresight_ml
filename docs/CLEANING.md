# Cleaning + Imputation (Processed Dataset)

This stage cleans the interim panel dataset and imputes missing values.

## Inputs (GCS)

- Interim panel parquet:
  - `gs://financial-distress-data/interim/panel_base.parquet`

## Outputs (GCS)

- Cleaned panel parquet:
  - `gs://financial-distress-data/processed/cleaned_panel.parquet`
- Cleaning report:
  - `gs://financial-distress-data/processed/cleaning_report.json`

## Default Policies

- Drop rows missing `cik` or `filing_date`
- Numeric: median per `cik`, fallback to global median
- Categorical: mode, fallback to `UNKNOWN`

## How to run locally

### 1) Credentials

Place your service account key at:

- `.gcp/service-account-key.json`

Export credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/.gcp/service-account-key.json"
```

### 2) Run cleaning

```bash
uv run python -m src.data.cleaning
```

### 3) Output files

- `data/processed/cleaned_panel.parquet`
- `data/processed/cleaning_report.json`
