# Preprocessing (Interim Dataset)

This stage prepares an interim, standardized panel dataset for downstream steps (cleaning, feature engineering, modeling).

## Inputs (GCS)

- SEC filings (JSONL):
  - `gs://financial-distress-data/raw/sec/year=2026/quarter=Q1/filings.jsonl`
- FRED indicators (CSV):
  - `gs://financial-distress-data/raw/fred/year=2026/month=02/indicators.csv`

## Output (GCS)

- Interim panel parquet:
  - `gs://financial-distress-data/interim/panel_base.parquet`

## How to run locally

### 1) Credentials

Place your service account key at:

- `.gcp/service-account-key.json`

Export credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/.gcp/service-account-key.json"
```
