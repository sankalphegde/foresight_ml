# Foresight-ML: Corporate Financial Distress Early-Warning System

**Foresight-ML** is an end-to-end MLOps initiative designed to predict corporate financial distress before it becomes irreversible. By leveraging historical financial data and machine learning, this system offers a dynamic alternative to static, lagging financial indicators.

> **📋 Quick Start**: See [SETUP.md](SETUP.md) for deployment
> **🔄 Data Collection**: 2020-2026 (6 years) | ~25 days to complete
> **🏗️ Architecture**: Single Airflow container runs all ingestion code

---

## 1. Project Description

The current landscape of corporate financial health monitoring suffers from inefficiencies that lead to "surprise" bankruptcies and delayed interventions. This project addresses two core problems:

* **Latency in Detection:** Financial distress is typically identified only after official quarterly reports (10-Q/10-K) are released. By the time a report is analyzed, the company may have been in distress for months.
* **Static & Outdated Thresholds:** Traditional methods rely on rigid rules (e.g., "Debt-to-Equity > 2.0"). These fail to adapt to changing macroeconomic conditions, such as shifting interest rate environments or industry-specific nuances.

**Foresight-ML** solves this by treating financial distress as a time-series classification problem, updating risk scores in near real-time as new market data becomes available.

---

## 2. Dataset Sources

This project utilizes a combination of fundamental and market data. The raw data is versioned using **GCS object versioning** to ensure reproducibility.
All data used in this project is **publicly available**, ensuring transparency, reproducibility, and suitability for academic research.

### Primary Data Sources
- **SEC EDGAR**: 10-K (annual) and 10-Q (quarterly) filings with structured XBRL financial statements
- **Federal Reserve Economic Data (FRED)**: Economic indicators including interest rates, inflation, credit spreads

### Data Management
- **Raw data**: Stored in Google Cloud Storage (GCS) with versioning enabled
- **Ingestion**: Orchestrated by Apache Airflow, executed directly in Airflow container
- **Storage paths**:
  - `raw/fred/year=X/month=Y/indicators.csv` - Economic indicators
  - `raw/sec/year=X/quarter=QY/filings.jsonl` - Company filings
  - `reference/companies.csv` - Company list (uploaded by Terraform)
- **Versioning**: GCS object versioning tracks all data changes automatically

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                        │
├─────────────────────────────────────────────────────────────┤
│  Apache Airflow (Docker Compose)                            │
│  ├── Scheduler: Orchestrates DAGs                           │
│  ├── Webserver: UI (http://localhost:8080)                  │
│  ├── Worker: Executes tasks                                 │
│  └── Database: Postgres metadata store                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ Triggers Cloud Run Jobs
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  GOOGLE CLOUD PLATFORM                      │
├─────────────────────────────────────────────────────────────┤
│  Cloud Run Jobs                                             │
│  ├── foresight-ingestion (FRED)                             │
│  │   └── Fetches economic indicators                        │
│  │       → Stores to GCS: raw/fred/                         │
│  │                                                          │
│  └── foresight-sec-ingestion (SEC)                          │
│      └── Fetches SEC filings                                │
│          → Stores to GCS: raw/sec/                          │
│                                                             │
│  Cloud Storage                                              │
│  └── financial-distress-data bucket                         │
│      ├── raw/fred/year=*/month=*/                           │
│      └── raw/sec/year=*/quarter=*/                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Quick Start (5 minutes)

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Git
- Google Cloud SDK (`gcloud` CLI)
- GCP Service Account with Cloud Run & Storage access

### Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd foresight-ml

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# 3. Install dependencies
uv sync

# 4. Copy environment file
cp example.env .env
# Edit .env with your API keys

# 5. Set up GCP credentials
gcloud auth application-default login
mkdir -p .gcp
gcloud iam service-accounts keys create .gcp/service-account-key.json \
  --iam-account=<your-service-account>@<project>.iam.gserviceaccount.com

# 6. Start Airflow
docker-compose up -d

# 7. Initialize Airflow
docker-compose run --rm airflow-webserver airflow db init
docker-compose run --rm airflow-webserver airflow users create \
  --username admin --firstname Admin --lastname User \
  --role Admin --email admin@foresight --password admin

# 8. Access UI
# Open http://localhost:8080 (admin / admin)
```

---

## 5. Full Setup Guide

### Step 1: Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd foresight-ml

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Install project dependencies
uv sync

# Copy and configure environment
cp example.env .env
# Edit .env and add your API keys:
# - GCP_PROJECT_ID
# - FRED_API_KEY
# - SEC_USER_AGENT
```

**Required environment variables** (in `.env`):
```env
GCP_PROJECT_ID=financial-distress-ew
GCP_REGION=us-central1
GCP_BUCKET_RAW=financial-distress-data
FRED_API_KEY=your-fred-api-key
SEC_USER_AGENT=your-user-agent
ENV=local
```

---

### Step 2: Google Cloud Setup

```bash
# Authenticate with GCP
gcloud auth application-default login
gcloud config set project financial-distress-ew

# Create service account for Airflow
gcloud iam service-accounts create foresight-airflow \
  --display-name="Foresight ML Airflow"

# Grant permissions
gcloud projects add-iam-policy-binding financial-distress-ew \
  --member="serviceAccount:foresight-airflow@financial-distress-ew.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

gcloud projects add-iam-policy-binding financial-distress-ew \
  --member="serviceAccount:foresight-airflow@financial-distress-ew.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Download service account key
mkdir -p .gcp
gcloud iam service-accounts keys create .gcp/service-account-key.json \
  --iam-account=foresight-airflow@financial-distress-ew.iam.gserviceaccount.com
```

---

### Step 3: Build Docker Images

Two separate images handle FRED and SEC ingestion:

#### Local Testing
```bash
# Build FRED image
docker build -f deployment/docker/Dockerfile.fred \
  -t us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest .

# Build SEC image
docker build -f deployment/docker/Dockerfile.sec \
  -t us-central1-docker.pkg.dev/financial-distress-ew/foresight/sec:latest .

# Test locally
docker run --env-file .env \
  us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest
```

#### Push to Google Artifact Registry
```bash
# Authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Create repository (one-time)
gcloud artifacts repositories create foresight \
  --repository-format=docker --location=us-central1

# Push images
docker push us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest
docker push us-central1-docker.pkg.dev/financial-distress-ew/foresight/sec:latest
```

#### Using Cloud Build (Automated)
```bash
# Submit build (automatically builds all three images)
gcloud builds submit --config deployment/cloudbuild.yaml --substitutions=_DOCKER_BUILDKIT=1

# Monitor build
gcloud builds log <BUILD_ID> --stream
```

---

### Step 4: Create Cloud Run Jobs

```bash
# FRED ingestion job
gcloud run jobs create foresight-ingestion \
  --image us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest \
  --region us-central1 \
  --memory 2Gi --cpu 2 --task-timeout 3600s \
  --set-env-vars EXECUTION_DATE="$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --set-env-vars GCS_BUCKET=financial-distress-data \
  --set-env-vars FRED_API_KEY=$(grep FRED_API_KEY .env | cut -d= -f2)

# SEC ingestion job
gcloud run jobs create foresight-sec-ingestion \
  --image us-central1-docker.pkg.dev/financial-distress-ew/foresight/sec:latest \
  --region us-central1 \
  --memory 2Gi --cpu 2 --task-timeout 3600s \
  --set-env-vars EXECUTION_DATE="$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --set-env-vars GCS_BUCKET=financial-distress-data \
  --set-env-vars SEC_USER_AGENT=$(grep SEC_USER_AGENT .env | cut -d= -f2)
```

**Test the jobs:**
```bash
# Execute FRED job
gcloud run jobs execute foresight-ingestion --region us-central1

# Execute SEC job
gcloud run jobs execute foresight-sec-ingestion --region us-central1

# View logs
gcloud run jobs logs read foresight-ingestion --region us-central1 --limit 50
gcloud run jobs logs read foresight-sec-ingestion --region us-central1 --limit 50
```

---

### Step 5: Set Up Local Airflow

**Start Airflow services:**
```bash
# Initialize Postgres database
docker-compose up postgres -d

# Initialize Airflow database
docker-compose run --rm airflow-webserver airflow db init

# Create admin user
docker-compose run --rm airflow-webserver airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@foresight.local \
  --password admin

# Start all services (webserver, scheduler, worker)
docker-compose up -d
```

**Verify services:**
```bash
docker-compose ps
```

**Access Airflow:**
- **URL:** http://localhost:8080
- **Username:** `admin`
- **Password:** `admin`

---

### Step 6: Configure and Run DAG

The DAG `foresight_ingestion` is automatically discovered and loaded.

**Trigger manually:**
1. Open http://localhost:8080
2. Find DAG: `foresight_ingestion`
3. Click "Trigger DAG"
4. Monitor execution in Airflow UI
5. Check Cloud Run job logs for details

**View task execution:**
- Click on DAG name to see task dependencies
- Click on task to view logs
- Check Cloud Run console for detailed job output

---

## 6. Docker Files Explained

### Dockerfile (FRED Ingestion)
```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Install dependencies (no project install)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-dev

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml uv.lock README.md ./

# Run FRED ingestion job
CMD ["uv", "run", "python", "-m", "src.ingestion.fred_job"]
```

### Dockerfile.sec (SEC Ingestion)
Same as Dockerfile but runs `src.ingestion.sec_job` instead.

---

## 7. Cloud Build Configuration

`deployment/cloudbuild.yaml` automatically builds all three images:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    # Builds Dockerfile → foresight/fred:latest

  - name: 'gcr.io/cloud-builders/docker'
    # Builds Dockerfile.sec → foresight/sec:latest

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/foresight/fred:$_IMAGE_TAG'
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/foresight/sec:$_IMAGE_TAG'
```

**Trigger build:**
```bash
gcloud builds submit --substitutions=_DOCKER_BUILDKIT=1
```

---

## 8. Airflow DAG Structure

**File:** `src/airflow/dags/foresight_ml_data_pipeline.py`

```python
with DAG(
    dag_id="foresight_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",  # Runs every Monday
    catchup=False,
) as dag:

    # Task 1: Fetch FRED data
    run_fred_ingestion = CloudRunExecuteJobOperator(
        task_id="run_fred_ingestion",
        job_name="foresight-ingestion",
    )

    # Task 2: Fetch SEC data (after FRED completes)
    run_sec_ingestion = CloudRunExecuteJobOperator(
        task_id="run_sec_ingestion",
        job_name="foresight-sec-ingestion",
    )

    # Define dependency: SEC runs after FRED
    run_fred_ingestion >> run_sec_ingestion
```

---

## 9. Project Structure

```
foresight-ml/
├── deployment/
│   ├── cloudbuild.yaml                 # Cloud Build configuration
│   └── docker/
│       ├── Dockerfile.fred             # FRED ingestion container
│       ├── Dockerfile.sec              # SEC ingestion container
│       └── Dockerfile.airflow          # Airflow orchestration container
├── docker-compose.yml                  # Local development stack
├── pyproject.toml                      # Python dependencies
├── uv.lock                             # Locked dependencies
├── example.env                         # Environment template
├── .env                                # Your config (gitignored)
├── .gcp/                               # GCP credentials (gitignored)
│   └── service-account-key.json
│
├── src/
│   ├── airflow/
│   │   └── dags/
│   │       └── foresight_ml_data_pipeline.py  # Orchestration DAG
│   │
│   ├── ingestion/                      # Data ingestion
│   │   ├── fred_job.py                 # Fetches FRED data
│   │   └── sec_job.py                  # Fetches SEC filings
│   │
│   ├── data/
│   │   ├── clients/
│   │   │   ├── fred_client.py          # FRED API wrapper
│   │   │   └── sec_client.py           # SEC API wrapper
│   │   ├── preprocess.py               # Data cleaning
│   │   └── split.py                    # Train/val/test splits
│   │
│   ├── feature_store/                  # Feature definitions
│   │   ├── definitions.py
│   │   └── repo.py
│   │
│   ├── models/                         # Model training
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── utils/
│       └── config.py
│
├── tests/                              # Unit tests
│   ├── test_data_ingestion.py
│   ├── test_data.py
│   └── test_model.py
│
├── notebooks/                          # Jupyter notebooks
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model_experiments.ipynb
│
├── infra/                              # Infrastructure (Terraform)
│   ├── orchestration/
│   │   └── composer.tf
│   └── iam/
│       └── iam.tf
│
├── monitoring/                         # Monitoring
│   ├── data_drift.py
│   ├── metrics.py
│   └── model_drift.py
│
└── README.md                           # This file
```

---

## 10. Common Tasks

### Run Ingestion Jobs Locally

```bash
# FRED
uv run python -m src.ingestion.fred_job

# SEC
uv run python -m src.ingestion.sec_job
```

### View Airflow Logs

```bash
# Scheduler logs
docker-compose logs -f airflow-scheduler

# Webserver logs
docker-compose logs -f airflow-webserver

# Worker logs
docker-compose logs -f airflow-worker
```

### Rebuild Images and Deploy

```bash
# Build both images
gcloud builds submit --config deployment/cloudbuild.yaml --substitutions=_DOCKER_BUILDKIT=1

# Update Cloud Run jobs
gcloud run jobs update foresight-ingestion \
  --image us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest \
  --region us-central1

gcloud run jobs update foresight-sec-ingestion \
  --image us-central1-docker.pkg.dev/financial-distress-ew/foresight/sec:latest \
  --region us-central1
```

### Schedule Jobs to Run Automatically

```bash
# Run every Monday at 2 AM UTC
gcloud run jobs update foresight-ingestion \
  --schedule "0 2 * * 1" \
  --region us-central1

gcloud run jobs update foresight-sec-ingestion \
  --schedule "0 3 * * 1" \
  --region us-central1
```

### Check GCS Data

```bash
gsutil ls gs://financial-distress-data/raw/
gsutil ls gs://financial-distress-data/raw/fred/
gsutil ls gs://financial-distress-data/raw/sec/
```

---

## 11. Troubleshooting

### Airflow DAG Not Running
```bash
# Check DAG syntax
docker-compose run --rm airflow-webserver airflow dags list

# Check scheduler logs
docker-compose logs airflow-scheduler

# Verify service account key
ls -la .gcp/service-account-key.json
```

### Cloud Run Job Fails
```bash
# View detailed logs
gcloud run jobs logs read <job-name> --region us-central1 --limit 100

# Check job configuration
gcloud run jobs describe <job-name> --region us-central1

# Test locally with Docker
docker run --env-file .env us-central1-docker.pkg.dev/financial-distress-ew/foresight/fred:latest
```

### Docker Build Issues
```bash
# Check Dockerfile syntax
docker build --progress=plain -f deployment/docker/Dockerfile.fred .

# Verify dependencies
uv export

# Check file permissions
ls -la deployment/docker/*.Dockerfile pyproject.toml uv.lock README.md
```

### Permission Errors
```bash
# Verify service account roles
gcloud projects get-iam-policy financial-distress-ew \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:foresight-airflow*"

# Test Cloud Run access
gcloud run jobs list --region us-central1
```

---

## 12. Environment Variables

### Required for Cloud Run Jobs

```env
# GCP
GCP_PROJECT_ID=financial-distress-ew
GCP_REGION=us-central1
GCP_BUCKET_RAW=financial-distress-data

# FRED
FRED_API_KEY=your-api-key-from-fred.stlouisfed.org

# SEC
SEC_USER_AGENT=your-email@example.com

# Execution
EXECUTION_DATE=2024-01-01T00:00:00  # Set automatically by Airflow
```

### Getting API Keys

**FRED:**
1. Visit https://fred.stlouisfed.org/docs/api/
2. Click "Request API Key"
3. Copy your key to `.env`

**SEC:**
- No key needed, but provide a User-Agent header
- Format: `app-name user@email.com`

---

## 13. Contributing

### Adding a New Ingestion Job

1. Create `src/ingestion/new_job.py`
2. Create corresponding Dockerfile
3. Update `deployment/cloudbuild.yaml` to build it
4. Create Cloud Run job
5. Add task to DAG
6. Test locally and in production

### Code Standards
- Run tests: `uv run pytest`
- Format code: `uv run ruff format src/`
- Lint: `uv run ruff check src/`
- Type check: `uv run mypy src/`

---

## 14. References

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar)
- [Docker Documentation](https://docs.docker.com/)
- [uv Package Manager](https://docs.astral.sh/uv/)

---

## 15. Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs (Airflow, Cloud Run, Docker)
3. Open an issue on GitHub
4. Contact the team lead

---

**Last Updated:** February 2026
