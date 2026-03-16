# Foresight-ML: MLOps Data Pipeline (Airflow DAG Submission)

End-to-end data pipeline for corporate financial distress modeling, built for the MLOps course requirements.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [DAG Flow](#3-current-dag-flow)
- [Project Structure](#4-project-structure-relevant-parts)
- [Local Setup](#5-local-setup-reproducible)
- [Run the Pipeline](#6-run-the-full-pipeline)
- [Outputs](#7-outputs-and-persistence)
- [Testing](#8-testing)
- [CI/CD](#cicd)
- [DVC Versioning](#9-dvc-data-versioning)
- [Schema / Anomaly Details](#10-schemastatisticsanomaly-details)
- [Bias Detection](#11-bias-detection-and-mitigation-notes)
- [Gantt / Optimization](#12-gantt-bottleneck-workflow-optimization)
- [MLflow Tracking](#mlflow-tracking)
- [Troubleshooting](#13-logging-and-troubleshooting)
- [Tech Stack](#15-tech-stack)

---

## Project Overview

This project builds a reproducible MLOps data pipeline that prepares corporate financial distress training data from raw public financial + macroeconomic signals.

Goals of this phase:
- Orchestrate the full workflow in Airflow DAG form
- Validate data quality and detect anomalies
- Generate feature/bias analysis outputs
- Keep data artifacts reproducible with DVC

---

## Data Sources

The pipeline uses two public sources:

- **SEC EDGAR (XBRL filings):** firm-level financial statement tags from periodic filings (10-Q/10-K style disclosures), ingested incrementally.
- **FRED (Federal Reserve Economic Data):** macro indicators (e.g., rates/inflation/labor proxies) ingested and aligned for downstream feature construction.

Source roles:
- SEC provides company-level fundamentals.
- FRED provides macro context.
- Both are combined during cleaning/feature stages for modeling and bias analysis.

---

## 1) What This Submission Includes

This repository contains a production-style DAG pipeline that covers:

- Data acquisition (SEC + FRED ingestion)
- Preprocessing and cleaning
- Panel and label generation
- Feature engineering + bias analysis
- Validation, schema/statistics summary, and anomaly detection
- Unit tests
- Data versioning with DVC
- Reproducible local execution with Docker + Airflow

---

## 2) Rubric Coverage

| Requirement | Status | Where Implemented |
|---|---|---|
| Data acquisition | Completed | `src/ingestion/fred_increment_job.py`, `src/ingestion/sec_xbrl_increment_job.py` |
| Data preprocessing | Completed | `src/data/cleaned/data_cleaned.sql`, `src/main_panel.py`, `src/main_labeling.py` |
| Test modules | Completed | `tests/` (ingestion, preprocessing, pipeline, feature, validation) |
| Airflow DAG orchestration | Completed | `src/airflow/dags/foresight_ml_data_pipeline.py` |
| DVC versioning | Completed | `data/final/final_v2.dvc`, `Makefile` DVC targets |
| Tracking/logging | Completed | Airflow task logs + Python logging in pipeline modules |
| Schema & statistics generation | Completed | `src/data/validate_anomalies.py` (`null_counts`, `null_rates`, `numeric_ranges`, required columns) |
| Anomaly detection & alerts | Completed | IQR anomaly detection + optional DAG fail flag `VALIDATION_FAIL_ON_STATUS=true` |
| Bias detection/data slicing | Completed | `src/feature_engineering/pipelines/bias_analysis.py`, feature/bias pipeline |
| Pipeline flow optimization (Gantt) | Completed | Airflow Gantt used; bottleneck task identified and mode switch added |
| Reproducibility | Completed | Local run steps + environment/config instructions below |
| Error handling | Completed | Environment checks, runtime checks, explicit task failures on invalid states |

---

## Architecture

```text
+-----------------------------------+
| Airflow DAG (foresight_ingestion) |
| daily orchestration               |
+---------------+-------------------+
                |
    +-----------+-----------+
    |                       |
    v                       v
+--------------------+    +--------------------+
| FRED ingestion     |    | SEC ingestion      |
| (incremental)      |    | (incremental/demo) |
+----------+---------+    +----------+---------+
            \                       /
             \                     /
              v                   v
          +-------------------------------+
          | GCS raw zone                  |
          | raw/fred/* , raw/sec_xbrl/*   |
          +---------------+---------------+
                          |
                          v
          +-------------------------------+
          | BigQuery cleaning SQL         |
          | cleaned_foresight.final_v2    |
          +---------------+---------------+
                          |
                          v
          +-------------------------------+
          | panel + labeling (GCS)        |
          | features/panel_v1, labeled_v1 |
          +---------------+---------------+
                          |
                          v
          +-------------------------------+
          | feature + bias pipeline       |
          | engineered_features (BQ)      |
          +---------------+---------------+
                          |
                          v
          +-------------------------------+
          | validation + anomaly          |
          | validation_report + anomalies |
          +-------------------------------+
```

---

## 3) Current DAG Flow

DAG ID: `foresight_ingestion`

Task order:

1. `run_fred_ingestion`
2. `run_sec_ingestion`
3. `run_preprocess_ingested_data`
4. `run_bigquery_cleaning`
5. `run_panel_build`
6. `run_labeling`
7. `run_feature_bias_pipeline`
8. `run_validation_anomaly`

### Feature/Bias Runtime Mode

The DAG supports an explicit mode switch:

- `FEATURE_BIAS_MODE=safe` (default): skips heavy visualizations for stable grading/demo runs
- `FEATURE_BIAS_MODE=full`: runs full visualization workload

Internally this controls `SKIP_HEAVY_VISUALIZATIONS` for the feature pipeline.

---

## 4) Project Structure (Relevant Parts)

```text
Foresight-ML/
├── src/
│   ├── airflow/dags/foresight_ml_data_pipeline.py
│   ├── ingestion/
│   │   ├── fred_increment_job.py
│   │   └── sec_xbrl_increment_job.py
│   ├── data/
│   │   ├── cleaned/data_cleaned.sql
│   │   └── validate_anomalies.py
│   ├── feature_engineering/
│   │   ├── pipelines/run_pipeline.py
│   │   ├── pipelines/feature_engineering.py
│   │   ├── pipelines/bias_analysis.py
│   │   └── config/settings.yaml
│   ├── main_panel.py
│   └── main_labeling.py
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_preprocess.py
│   ├── test_pipeline.py
│   ├── test_validation.py
│   └── test_feature_engineering/
├── infra/
├── deployment/docker/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── data/final/final_v2.dvc
```

### Legacy Modules (Retained, Non-Destructive)

The following ingestion modules are kept for backward compatibility and historical reference:

- `src/ingestion/fred_job.py`
- `src/ingestion/sec_job.py`

Current Airflow execution uses the incremental modules (`fred_increment_job.py`, `sec_xbrl_increment_job.py`). The active orchestration path is defined in `src/airflow/dags/foresight_ml_data_pipeline.py`.

---

## 5) Local Setup (Reproducible)

### Prerequisites

- Python 3.12+
- Docker Desktop
- Access to GCP project + service account key for local run

### Environment

```bash
cp .env.example .env
```

Minimum required keys:

```bash
GCP_PROJECT_ID=financial-distress-ew
GCS_BUCKET=financial-distress-data
FRED_API_KEY=<your-fred-key>
SEC_USER_AGENT="foresight-ml your-email@example.com"
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/.gcp/foresight-data-sa.json

# Optional behavior flags
FEATURE_BIAS_MODE=safe
VALIDATION_FAIL_ON_STATUS=false
GCS_VALIDATION_REPORT_OUT=processed/validation_report.json
GCS_ANOMALIES_OUT=processed/anomalies.parquet
```

Secrets handling:

- **Local development:** values read from `.env` (never commit real keys)
- **CI:** injected from GitHub Actions secrets
- **CD (Cloud Run jobs):** credentials injected via `gcloud run deploy --set-secrets`
- **Terraform-managed Airflow:** env injected via Terraform variables

### Install

```bash
make setup
```

### Start Local Airflow

```bash
make local-up
```

Airflow UI: `http://localhost:8080` — username: `admin` / password: `admin`

### Stop

```bash
make local-down
```

---

## 6) Run the Full Pipeline

Trigger from the Airflow UI or CLI:

```bash
docker compose exec airflow airflow dags unpause foresight_ingestion
docker compose exec airflow airflow dags trigger foresight_ingestion --run-id manual_submission_$(date +%Y%m%d_%H%M%S)
```

Check task states:

```bash
docker compose exec airflow airflow tasks states-for-dag-run foresight_ingestion <RUN_ID>
```

---

## 7) Outputs and Persistence

### GCS

| Path | Description |
|---|---|
| `raw/sec_xbrl/cik=<cik>/data.parquet` | Raw SEC filings |
| `raw/fred/series_id=<id>.parquet` | Raw FRED series |
| `cleaned_data/final_v2/train_*.parquet` | Cleaned export from SQL |
| `features/panel_v1/panel.parquet` | Panel output |
| `features/labeled_v1/labeled_panel.parquet` | Labeled panel |
| `processed/validation_report.json` | Validation report |
| `processed/anomalies.parquet` | Anomaly rows |

### BigQuery

- `cleaned_foresight.final_v2`
- `financial_distress_features.engineered_features`
- `financial_distress_features.cleaned_engineered_features`

---

## 8) Testing

Run all tests:

```bash
make test
```

Run targeted suites:

```bash
# Ingestion
uv run pytest tests/test_data_ingestion.py -q

# Preprocessing / cleaning
uv run pytest tests/test_preprocess.py tests/test_cleaned.py -q

# Pipeline / model
uv run pytest tests/test_pipeline.py tests/test_model.py tests/test_data.py -q

# Validation
uv run pytest tests/test_validation.py -q

# Feature engineering
uv run pytest tests/test_feature_engineering -q
```

Key test modules:

- `tests/test_data_ingestion.py`
- `tests/test_preprocess.py`
- `tests/test_cleaned.py`
- `tests/test_pipeline.py`
- `tests/test_model.py`
- `tests/test_data.py`
- `tests/test_validation.py`
- `tests/test_feature_engineering/test_feature_engineering.py`
- `tests/test_feature_engineering/test_bias_analysis.py`

---

## CI/CD

GitHub Actions workflows under `.github/workflows/`:

- `ci.yml`: linting, typing, tests, and validation checks
- `cd-dev.yml`: deployment pipeline for dev environment

Typical CI checks:

- Ruff lint/style
- mypy static type checks
- pytest execution (including validation module)
- Terraform format/validate

---

## 9) DVC Data Versioning

Initialize and configure DVC remote:

```bash
make dvc-setup
```

Push/pull tracked data:

```bash
make dvc-push
make dvc-pull
```

Track final dataset from GCS URI:

```bash
export FINAL_DATASET_GCS_URI=gs://<bucket>/<path>/final_dataset.parquet
make dvc-track-final
```

---

## 10) Schema/Statistics/Anomaly Details

`src/data/validate_anomalies.py` generates:

- Required column checks (`cik`, `filing_date`, `ticker`, `accession_number`)
- Duplicate count on (`cik`, `accession_number`)
- Null counts and null rates by column
- Numeric min/max ranges by column
- IQR-based anomaly rows + per-column anomaly counts

Alert behavior:

- `VALIDATION_FAIL_ON_STATUS=true`: DAG task fails when validation status is `fail`
- `VALIDATION_FAIL_ON_STATUS=false`: uploads artifacts and logs status for downstream review

---

## 11) Bias Detection and Mitigation Notes

Bias analysis is implemented in the feature pipeline:

- Data slicing across company size, sector proxy, time split, macro regime, and distress label
- Drift metrics and high-drift alerts (PSI)
- Slice-level summaries and fairness diagnostics

Mitigation strategy is documented in the generated bias report (recommendations include class imbalance handling, drift investigation, and stratified evaluation).

---

## 12) Gantt Bottleneck / Workflow Optimization

1. Open Airflow UI → run details → **Gantt** view
2. Capture screenshot and analyze longest tasks

Recent bottleneck identified: `run_feature_bias_pipeline`

Optimization applied: `FEATURE_BIAS_MODE=safe` default reduces heavy plotting risk and improves run reliability.

![Pipeline Gantt Chart](docs/images/pipeline_gantt.png)

---

## MLflow Tracking

MLflow infrastructure is deployed but model training/evaluation runtime is not yet active.

Deployment shape:

- **Tracking server:** Cloud Run service (`foresight-mlflow`)
- **Metadata backend:** Cloud SQL PostgreSQL
- **Artifacts:** `gs://$GCS_BUCKET/mlflow/artifacts`

### Deploy MLflow Infra

```bash
source .env
cd infra
terraform apply

export MLFLOW_TRACKING_URI="$(terraform output -raw mlflow_tracking_uri)"
```

### Intentionally Stubbed Files

The following are scaffolded placeholders for future implementation:

- `src/models/train.py`
- `src/models/evaluate.py`
- `src/models/predict.py`

These currently raise `NotImplementedError`. When ready:

1. Add training + metric logging in `src/models/train.py`
2. Add evaluation logging in `src/models/evaluate.py`
3. Add model loading/inference in `src/models/predict.py`

Optional connectivity check:

```bash
source .env
curl -I "$MLFLOW_TRACKING_URI"
```

---

## 13) Logging and Troubleshooting

Airflow logs are available per task in the UI and container logs. Typical failure guards are included for missing env vars, missing SQL/config files, and validation status failures.

Useful commands:

```bash
docker compose ps
docker compose logs airflow --since 15m
docker compose exec airflow airflow tasks list foresight_ingestion
```

---

## 14) Notes

- Use `FEATURE_BIAS_MODE=safe` for reliable full DAG execution on limited local resources.
- Use `FEATURE_BIAS_MODE=full` if full visualization artifacts are required and resources are sufficient.
- Validation is intentionally placed after the feature+bias stage to match the current design.

---

## 15) Tech Stack

- Python 3.12
- Apache Airflow (local image: `apache/airflow:slim-3.1.7-python3.12`)
- Google Cloud Storage + BigQuery
- Docker Compose
- Terraform
- DVC
- pytest, mypy, ruff

---

## 16) Scalability Considerations

The DAG is structured with independent, modular tasks and parallel ingestion branches, allowing horizontal scaling strategies such as partitioned or entity-level ingestion if required in larger production environments.

---

Last updated: March 2026
