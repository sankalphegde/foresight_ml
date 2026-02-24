# Foresight-ML: MLOps Data Pipeline (Airflow DAG Submission)

End-to-end data pipeline for corporate financial distress modeling, built for the MLOps course requirements.

## Project overview

This project builds a reproducible MLOps data pipeline that prepares corporate financial distress training data from raw public financial + macroeconomic signals.

Goal of this phase:
- orchestrate the full workflow in Airflow DAG form,
- validate data quality and detect anomalies,
- generate feature/bias analysis outputs,
- and keep data artifacts reproducible with DVC.

---

## Data sources

The pipeline uses two public sources:
- **SEC EDGAR (XBRL filings):** firm-level financial statement tags from periodic filings (10-Q/10-K style disclosures), ingested incrementally.
- **FRED (Federal Reserve Economic Data):** macro indicators (e.g., rates/inflation/labor proxies) ingested and aligned for downstream feature construction.

Source roles in pipeline:
- SEC provides company-level fundamentals.
- FRED provides macro context.
- Both are combined during cleaning/feature stages for modeling and bias analysis.

---

## 1) What this submission includes

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

## 2) Rubric coverage (quick checklist)

| Requirement | Status | Where implemented |
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
	 +--------------------+    +---------------------+
	 | FRED ingestion      |   | SEC ingestion       |
	 | (incremental)       |   | (incremental/demo)  |
	 +----------+----------+   +----------+----------+
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

## 3) Current DAG flow

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

### Feature/Bias runtime mode

The DAG supports an explicit mode switch:
- `FEATURE_BIAS_MODE=safe` (default): skips heavy visualizations for stable grading/demo runs
- `FEATURE_BIAS_MODE=full`: runs full visualization workload

Internally this controls `SKIP_HEAVY_VISUALIZATIONS` for the feature pipeline.

---

## 4) Project structure (relevant parts)

```text
Foresight-ML/
├─ src/
│  ├─ airflow/dags/foresight_ml_data_pipeline.py
│  ├─ ingestion/
│  │  ├─ fred_increment_job.py
│  │  └─ sec_xbrl_increment_job.py
│  ├─ data/
│  │  ├─ cleaned/data_cleaned.sql
│  │  └─ validate_anomalies.py
│  ├─ feature_engineering/
│  │  ├─ pipelines/run_pipeline.py
│  │  ├─ pipelines/feature_engineering.py
│  │  ├─ pipelines/bias_analysis.py
│  │  └─ config/settings.yaml
│  ├─ main_panel.py
│  └─ main_labeling.py
├─ tests/
│  ├─ test_data_ingestion.py
│  ├─ test_preprocess.py
│  ├─ test_pipeline.py
│  ├─ test_validation.py
│  └─ test_feature_engineering/
├─ infra/
├─ deployment/docker/
├─ docker-compose.yml
├─ Makefile
├─ pyproject.toml
└─ data/final/final_v2.dvc
```

---

## 5) Local setup (reproducible)

### Prerequisites
- Python 3.12+
- Docker Desktop
- Access to GCP project + service account key for local run

### Environment
Create a `.env` file in repo root with at least:

```bash
GCP_PROJECT_ID=financial-distress-ew
GCS_BUCKET=financial-distress-data
FRED_API_KEY=<your-fred-key>
SEC_USER_AGENT="foresight-ml your-email@example.com"
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/.gcp/foresight-data-sa.json

# Optional stability/behavior flags
FEATURE_BIAS_MODE=safe
VALIDATION_FAIL_ON_STATUS=false
GCS_VALIDATION_REPORT_OUT=processed/validation_report.json
GCS_ANOMALIES_OUT=processed/anomalies.parquet
```

### Install + quality tools

```bash
make setup
```

### Start local Airflow

```bash
make local-up
```

UI:
- URL: `http://localhost:8080`
- Username: `admin`
- Password: `admin123`

Stop:

```bash
make local-down
```

---

## 6) Run the full pipeline

Trigger from Airflow UI or CLI:

```bash
docker compose exec airflow airflow dags unpause foresight_ingestion
docker compose exec airflow airflow dags trigger foresight_ingestion --run-id manual_submission_$(date +%Y%m%d_%H%M%S)
```

Check task states:

```bash
docker compose exec airflow airflow tasks states-for-dag-run foresight_ingestion <RUN_ID>
```

---

## 7) Outputs and persistence

### GCS
- Raw SEC: `raw/sec_xbrl/cik=<cik>/data.parquet`
- Raw FRED: `raw/fred/series_id=<id>.parquet`
- Cleaned export (from SQL): `cleaned_data/final_v2/train_*.parquet`
- Panel: `features/panel_v1/panel.parquet`
- Labeled: `features/labeled_v1/labeled_panel.parquet`
- Validation report: `processed/validation_report.json`
- Anomalies: `processed/anomalies.parquet`

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

Run the full suite directly:

```bash
uv run pytest tests/ -q
```

Run ingestion tests:

```bash
uv run pytest tests/test_data_ingestion.py -q
```

Run preprocessing/cleaning tests:

```bash
uv run pytest tests/test_preprocess.py tests/test_cleaned.py -q
```

Run pipeline/model tests:

```bash
uv run pytest tests/test_pipeline.py tests/test_model.py tests/test_data.py -q
```

Run validation tests only:

```bash
uv run pytest tests/test_validation.py -q
```

Run feature engineering tests:

```bash
uv run pytest tests/test_feature_engineering -q
```

Current key test modules include:
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

GitHub Actions workflows are included under `.github/workflows/`:
- `ci.yml`: quality gates for linting, typing, tests, and validation checks
- `cd-dev.yml`: deployment pipeline for dev environment updates

Typical CI checks include:
- Ruff lint/style checks
- mypy static type checks
- pytest test execution (including validation module tests)
- Terraform formatting/validation where configured

This ensures code quality and reproducibility before deployment.

---

## 9) DVC data versioning

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

## 10) Schema/statistics/anomaly details

`src/data/validate_anomalies.py` generates:
- Required column checks (`cik`, `filing_date`, `ticker`, `accession_number`)
- Duplicate count on (`cik`, `accession_number`)
- Null counts and null rates by column
- Numeric min/max ranges by column
- IQR-based anomaly rows + per-column anomaly counts

Alert behavior:
- If `VALIDATION_FAIL_ON_STATUS=true`, DAG task fails when validation status is `fail`.
- If false, task uploads artifacts and logs status while allowing downstream review.

---

## 11) Bias detection and mitigation notes

Bias analysis is implemented in the feature pipeline:
- Data slicing across company size, sector proxy, time split, macro regime, and distress label
- Drift metrics and high-drift alerts (PSI)
- Slice-level summaries and fairness diagnostics

Mitigation strategy documented in generated bias report (recommendations include class imbalance handling, drift investigation, and stratified evaluation).

---

## 12) Gantt bottleneck workflow optimization

For course submission:
1. Open Airflow UI run details
2. Go to **Gantt** view
3. Capture screenshot
4. Analyze longest tasks

Recent bottleneck identified:
- `run_feature_bias_pipeline` (historically longest)

Optimization applied:
- Added `FEATURE_BIAS_MODE=safe` default to reduce heavy plotting risk and improve run reliability.

---

## 13) Logging and troubleshooting

- Airflow logs are available per task in UI and container logs.
- Typical failure guards included for missing env vars, missing SQL/config files, and validation status failures.

Useful commands:

```bash
docker compose ps
docker compose logs airflow --since 15m
docker compose exec airflow airflow tasks list foresight_ingestion
```

---

## 14) Notes for graders

- Use `FEATURE_BIAS_MODE=safe` for reliable full DAG execution on limited local resources.
- Use `FEATURE_BIAS_MODE=full` if full visualization artifacts are required and resources are sufficient.
- Validation is intentionally placed after feature+bias stage to match current design.

---

## 15) Tech stack

- Python 3.12
- Apache Airflow (local image: `apache/airflow:slim-3.1.7-python3.12`)
- Google Cloud Storage + BigQuery
- Docker Compose
- Terraform
- DVC
- pytest, mypy, ruff

---

Last updated: February 2026
