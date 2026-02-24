# Foresight-ML: Corporate Financial Distress Early-Warning System

End-to-end MLOps pipeline for predicting corporate financial distress using public SEC filings and economic indicators.

> **🏗️ Stack**: Airflow on Cloud Run + GCS + BigQuery + DVC
> **📊 Data**: 2020-2026 SEC/FRED data with DVC versioning
> **⚡ Deploy**: `terraform apply` (single command)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [DVC Setup](#dvc-setup)
- [Development](#development)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Common Tasks](#common-tasks)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## Overview

MLOps pipeline for predicting corporate financial distress using:
- **SEC filings**: 10-K/10-Q financial statements
- **Economic indicators**: FRED data (interest rates, inflation, credit spreads)
- **ML models**: Time-series classification with daily updates

---

## Architecture

```
┌────────────────────────────┐
│  Airflow (Cloud Run)       │
│  ├─ FRED ingestion         │
│  └─ SEC ingestion          │
└───────────┬────────────────┘
           │
           ▼
    ┌───────────────┐
    │ GCS + BigQuery│
    └──────┬────────┘
           │
           ▼
    ┌────────────┐
    │ Local Dev  │
    │ (DVC)      │
    └────────────┘
```

- **Airflow (Cloud Run)**: Orchestrates data ingestion
- **GCS**: Stores raw data and DVC remote storage
- **BigQuery**: Structured data warehouse
- **DVC**: Version control for data and models (local dev only)
- **Terraform**: Infrastructure as code

### Data Flow

1. **Ingestion**: Airflow DAG runs daily, fetches data from SEC/FRED APIs
2. **Storage**: Raw data → GCS with partitioning (year/quarter)
3. **Analytics**: Data loaded into BigQuery
4. **Local Development**: Pull data → process/experiment → track with DVC

---

## Quick Start

### Prerequisites

- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **Terraform 1.6+** ([install](https://developer.hashicorp.com/terraform/downloads))
- **Docker** ([install](https://docs.docker.com/get-docker/))
- **GCP account** with billing enabled
- **FRED API key** ([get key](https://fred.stlouisfed.org/docs/api/))

### 1. Clone and Configure

```bash
git clone <repo-url>
cd foresight-ml

# Copy and edit environment file
cp example.env .env
# Edit .env with:
#   - GCP_PROJECT_ID
#   - FRED_API_KEY
#   - SEC_USER_AGENT

source .env
```

### 2. Authenticate with GCP

```bash
# Login and set project
gcloud auth login
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
gcloud services enable \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com
```

### 3. Deploy Infrastructure

```bash
cd infra
terraform init
terraform plan    # Review what will be created
terraform apply   # Type 'yes' to confirm

# Get Airflow URL
terraform output airflow_url
```

**What gets created:**
- GCS bucket: `{project-id}-foresight-ml-data`
- BigQuery dataset: `foresight_ml_dev`
- Artifact Registry repository
- Cloud Run service (Airflow)
- IAM service accounts with minimal permissions

### 4. Access Airflow

```bash
# Get URL
terraform output airflow_url

# Login: admin / admin
# Enable the DAG: foresight_ml_data_pipeline
```

---

## Data Sources

All data is publicly available:

### SEC EDGAR
- **Content**: 10-K/10-Q filings with XBRL financial statements
- **Companies**: S&P 500 constituents
- **Frequency**: Quarterly/Annual
- **Storage**: `gs://.../raw/sec/year=YYYY/quarter=Q/`
- **API**: https://www.sec.gov/edgar/sec-api-documentation

### FRED (Federal Reserve Economic Data)
- **Indicators**:
  - Interest rates (10-year Treasury, Fed Funds)
  - Inflation (CPI, PCE)
  - Credit spreads (BAA-AAA)
- **Frequency**: Daily/Monthly
- **Storage**: `gs://.../raw/fred/year=YYYY/month=MM/`
- **API**: https://fred.stlouisfed.org/docs/api/

### Data Timeline
- **Historical**: 2020-01-01 to present (~6 years)
- **Ingestion**: Daily with catchup enabled
- **Backfill**: ~25 days to complete initial load

---

## Project Structure

```
foresight_ml/
├── infra/                          # Terraform infrastructure
│   ├── artifact_registry.tf        # Docker registry + Cloud Build
│   ├── bigquery.tf                 # BigQuery dataset
│   ├── cloud_run.tf                # Airflow deployment
│   ├── storage.tf                  # GCS buckets
│   ├── iam.tf                      # Service accounts
│   └── variables.tf
│
├── src/
│   ├── airflow/dags/
│   │   └── foresight_ml_data_pipeline.py  # Main orchestration DAG
│   ├── ingestion/
│   │   ├── fred_job.py             # FRED data fetcher
│   │   └── sec_job.py              # SEC filing fetcher
│   ├── data/
│   │   ├── clients/                # API client wrappers
│   │   ├── preprocess.py           # Data cleaning
│   │   └── split.py                # Train/test split
│   ├── models/
│   │   ├── train.py                # Model training
│   │   ├── evaluate.py             # Model evaluation
│   │   └── predict.py              # Inference
│   └── utils/
│       └── config.py               # Configuration management
│
├── deployment/
│   ├── cloudbuild.yaml             # Automated Docker builds
│   └── docker/
│       └── Dockerfile.airflow      # Airflow container
│
├── tests/                          # Unit + integration tests
├── notebooks/                      # EDA + experiments
├── monitoring/                     # Drift detection
├── data/                           # Local data (gitignored, DVC tracked)
├── cache/                          # API response cache
│
├── Makefile                        # Development commands
├── pyproject.toml                  # Python dependencies (uv)
├── docker-compose.yml              # Local Airflow
└── README.md
```

---

## DVC Setup

DVC tracks data and model versions locally. Use it for experiments, processed datasets, and model artifacts.

### Initial Setup

```bash
# Install dependencies
make setup

# Load environment
source .env

# Initialize DVC with GCS remote
make dvc-setup
```

### Tracking Data

```bash
# Track a file or directory
uv run dvc add data/companies.csv
uv run dvc add data/processed/

# If the final dataset lives in GCS, track it via DVC import
export FINAL_DATASET_GCS_URI=gs://<bucket>/<path>/final_dataset.parquet
make dvc-track-final

# Commit .dvc metadata to Git
git add data/companies.csv.dvc data/processed.dvc .gitignore
git commit -m "Track data with DVC"

# Push actual data to GCS
make dvc-push
```

For imported datasets, commit the generated metadata files:

```bash
git add data/final/final_dataset.parquet.dvc dvc.lock .gitignore
git commit -m "Track final dataset from GCS with DVC"
```

### Collaborating with DVC

```bash
# On another machine, after git clone:
make setup
source .env
make dvc-setup
make dvc-pull  # Downloads all tracked data from GCS

# Check what changed
uv run dvc status

# After making changes
uv run dvc add data/processed/
git add data/processed.dvc
git commit -m "Update processed data"
make dvc-push
```

---

## Development

### Local Setup

```bash
# Install uv, dependencies, pre-commit hooks
make setup

# Start local Airflow
make local-up

# Access Airflow UI
open http://localhost:8080  # admin/admin

# Stop Airflow
make local-down
```

### Code Quality

```bash
# Run all checks (format + lint + type check + terraform)
make check

# Individual commands
make format          # Auto-format with ruff
make lint            # Check code style
make typecheck       # Run mypy
make terraform-check # Validate Terraform

# Run tests
make test

# Run specific test
uv run pytest tests/test_data_ingestion.py -v -k test_fred
```

### Running Jobs Locally

```bash
# Load API keys
source .env

# Run FRED ingestion
uv run python -m src.ingestion.fred_job

# Run SEC ingestion
uv run python -m src.ingestion.sec_job
```

---

## CI/CD

### GitHub Actions

Workflow file: `.github/workflows/ci.yml`

**On every push:**
- Code formatting check (ruff)
- Linting (ruff)
- Type checking (mypy)
- Terraform validation
- Unit tests
- Integration tests (with API keys)

### Required Secrets

Configure in **GitHub Settings → Secrets → Actions**:

| Secret | Value |
|--------|-------|
| `FRED_API_KEY` | Your FRED API key |
| `SEC_USER_AGENT` | `app-name your@email.com` |

### Deployment

Infrastructure changes are deployed manually via Terraform:

```bash
cd infra
terraform plan
terraform apply
```

For code changes, rebuild the Docker image:

```bash
cd infra
# Terraform will detect Dockerfile changes and rebuild
terraform apply
```

---

## Testing

### Run All Tests

```bash
make test
```

### Unit Tests Only (No API Calls)

```bash
FRED_API_KEY="" SEC_USER_AGENT="" make test
```

### Integration Tests (Requires API Keys)

```bash
source .env
uv run pytest tests/test_data_ingestion.py -v
```

### Test Coverage

```bash
uv run pytest --cov=src tests/
```

---

## Common Tasks

### View Logs

```bash
# Cloud Run logs
gcloud run services logs tail foresight-airflow --region us-central1

# Follow logs
gcloud run services logs tail foresight-airflow --region us-central1 --follow
```

### Check Data in GCS

```bash
# List buckets
gsutil ls

# Check FRED data
gsutil ls gs://$GCS_BUCKET/raw/fred/

# Check SEC data
gsutil ls gs://$GCS_BUCKET/raw/sec/

# Download a file
gsutil cp gs://$GCS_BUCKET/raw/fred/year=2024/month=01/data.json .
```

### Query BigQuery

```bash
# List datasets
bq ls

# Query data
bq query --use_legacy_sql=false \
  "SELECT * FROM \`$GCP_PROJECT_ID.foresight_ml_dev.raw_filings\` LIMIT 10"
```

### Update Infrastructure

```bash
cd infra
terraform plan
terraform apply
```

### Force Rebuild Docker Image

Edit `force_rebuild` in `infra/artifact_registry.tf` or just touch the Dockerfile:

```bash
touch deployment/docker/Dockerfile.airflow
cd infra && terraform apply
```

### Cleanup

```bash
cd infra
terraform destroy  # Type 'yes' to confirm
```

---

## Configuration

### Environment Variables

Required in `.env`:

```bash
# GCP Configuration
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET="${GCP_PROJECT_ID}-foresight-ml-data"

# API Keys
export FRED_API_KEY="your-fred-api-key"
export SEC_USER_AGENT="foresight-ml your-email@example.com"

# Terraform Variables
export TF_VAR_project_id="${GCP_PROJECT_ID}"
export TF_VAR_fred_api_key="${FRED_API_KEY}"
export TF_VAR_sec_user_agent="${SEC_USER_AGENT}"
export TF_VAR_region="us-central1"
```

### Getting API Keys

**FRED:**
1. Visit https://fred.stlouisfed.org/docs/api/
2. Sign in/create account
3. Click "Request API Key"
4. Add to `.env`: `FRED_API_KEY=your-key-here`

**SEC:**
- No API key required
- Must provide User-Agent with contact email
- Format: `SEC_USER_AGENT="app-name your@email.com"`

---

## Troubleshooting

### Terraform Issues

**Permission Denied:**
```bash
# Re-authenticate with full scope
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform

# Check project
gcloud config get-value project
```

**Resource Already Exists:**
```bash
# Import existing resource
cd infra
terraform import google_storage_bucket.data_lake $GCP_PROJECT_ID-foresight-ml-data
```

### Airflow Not Starting

```bash
# Check Cloud Run status
gcloud run services describe foresight-airflow --region us-central1

# View logs
gcloud run services logs tail foresight-airflow --region us-central1
```

### API Rate Limits

**FRED:** 120 requests/minute
**SEC:** 10 requests/second

Adjust `sleep()` calls in ingestion jobs if hitting limits.

### DVC Issues

```bash
# Re-configure remote
uv run dvc remote modify storage credentialpath $GOOGLE_APPLICATION_CREDENTIALS

# Check status
uv run dvc status

# Force push
uv run dvc push --force
```

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow | 2.8+ |
| **Cloud Platform** | Google Cloud Platform | - |
| **Compute** | Cloud Run | - |
| **Storage** | GCS, BigQuery | - |
| **Infrastructure** | Terraform | 1.6+ |
| **Language** | Python | 3.12 |
| **Package Manager** | uv | 0.5+ |
| **Data Versioning** | DVC | 3.66+ |
| **Containerization** | Docker | - |
| **CI/CD** | GitHub Actions | - |
| **Linting** | Ruff | - |
| **Type Checking** | mypy | - |
| **Testing** | pytest | - |

---

## References

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [SEC EDGAR API](https://www.sec.gov/edgar/sec-api-documentation)
- [DVC Documentation](https://dvc.org/doc)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [uv Package Manager](https://docs.astral.sh/uv/)

---

**Last Updated:** February 2026
