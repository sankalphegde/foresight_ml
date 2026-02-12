# Foresight-ML: Corporate Financial Distress Early-Warning System

**Foresight-ML** is an end-to-end MLOps pipeline designed to predict corporate financial distress before it becomes irreversible. By leveraging historical financial data and machine learning, this system offers a dynamic alternative to static, lagging financial indicators.

> **📋 Quick Start**: See [SETUP.md](SETUP.md) for deployment instructions
> **🔄 Data Collection**: 2020-2026 (6 years historical data)
> **🏗️ Architecture**: Airflow on Cloud Run + GCS + BigQuery
> **⚡ Deployment**: `terraform apply` (single command)

---

## Overview

Traditional financial health monitoring suffers from two critical inefficiencies:

1. **Latency in Detection**: Financial distress is typically identified only after official quarterly reports (10-Q/10-K) are released—often months after problems begin
2. **Static & Outdated Thresholds**: Traditional methods rely on rigid rules (e.g., "Debt-to-Equity > 2.0") that fail to adapt to changing macroeconomic conditions

**Foresight-ML** addresses these by treating financial distress as a time-series classification problem, updating risk scores as new data becomes available.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Airflow on Cloud Run                   │
│  ┌─────────────────────────────────┐    │
│  │ DAG: foresight_ingestion        │    │
│  │  ├─ FRED ingestion (Python)     │    │
│  │  └─ SEC ingestion (Python)      │    │
│  └─────────────────────────────────┘    │
└────────────┬────────────────────────────┘
             │
             ▼
      ┌─────────────┐
      │  GCS Bucket │  ← Raw data storage
      │  BigQuery   │  ← Analytics/queries
      └─────────────┘
```

**Key Components:**
- **Airflow on Cloud Run**: Containerized Airflow deployed via Terraform, orchestrates data ingestion
- **Data Sources**:
  - SEC EDGAR (10-K/10-Q filings with XBRL)
  - FRED (economic indicators: interest rates, inflation, credit spreads)
- **Storage**: GCS with versioning for raw data, BigQuery for structured analytics
- **Orchestration**: Daily ingestion with 6-year historical backfill (~25 days)

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Terraform 1.6+**
- **Docker** (for local development)
- **GCP Account** with billing enabled

### Setup Authentication

**For Terraform Deployment:**
```bash
# Use your personal account with Owner permissions
gcloud auth login
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform
gcloud config set project financial-distress-ew

# Do NOT set GOOGLE_APPLICATION_CREDENTIALS for Terraform
unset GOOGLE_APPLICATION_CREDENTIALS
```

**For Local Development (Optional - only needed to run ingestion scripts locally):**
```bash
# Authenticate and set project
gcloud auth login
gcloud config set project financial-distress-ew

# Create service account (skip if exists)
gcloud iam service-accounts create foresight-ml-sa \
  --display-name="Foresight ML" 2>/dev/null || echo "Service account already exists"

# Grant permissions
gcloud projects add-iam-policy-binding financial-distress-ew \
  --member="serviceAccount:foresight-ml-sa@financial-distress-ew.iam.gserviceaccount.com" \
  --role="roles/storage.admin" --condition=None

# Download key
mkdir -p .gcp
gcloud iam service-accounts keys create .gcp/service-account-key.json \
  --iam-account=foresight-ml-sa@financial-distress-ew.iam.gserviceaccount.com
```

### Deploy Infrastructure

```bash
# 1. Clone and setup
git clone <repo-url>
cd foresight-ml
cp example.env .env
# Edit .env with your GCP_PROJECT_ID and API keys

# 2. Load environment
source .env

# 3. Deploy with Terraform
cd infra
terraform init
terraform plan
terraform apply

# 4. Get Airflow URL
terraform output airflow_url
# Login: admin / admin
```

That's it! Terraform handles:
- GCS buckets + versioning
- BigQuery dataset
- Artifact Registry
- Docker image build via Cloud Build
- Cloud Run service deployment
- IAM service accounts
- Company list upload

**See [SETUP.md](SETUP.md) for detailed setup instructions including local development.**

---

## Data Sources

All data is **publicly available** for transparency and reproducibility:

| Source | Data Type | Frequency | Storage Path |
|--------|-----------|-----------|--------------|
| **SEC EDGAR** | 10-K/10-Q filings, XBRL financials | Quarterly/Annual | `gs://.../raw/sec/year=*/quarter=*/` |
| **FRED** | Interest rates, inflation, credit spreads | Daily/Monthly | `gs://.../raw/fred/year=*/month=*/` |
| **Reference** | S&P 500 company list | Static | `gs://.../reference/companies.csv` |

**Data Management:**
- GCS object versioning tracks all changes automatically
- BigQuery tables for structured queries
- Partitioned by date for efficient queries

---

## Project Structure

```
foresight_ml/
├── infra/                          # Infrastructure as Code (Terraform)
│   ├── artifact_registry.tf        # Docker registry + Cloud Build
│   ├── bigquery.tf                 # BigQuery dataset
│   ├── cloud_run.tf                # Airflow deployment
│   ├── storage.tf                  # GCS buckets
│   ├── iam.tf                      # Service accounts + IAM
│   ├── outputs.tf
│   ├── providers.tf
│   └── variables.tf
│
├── src/
│   ├── airflow/dags/
│   │   └── foresight_ml_data_pipeline.py  # Orchestration DAG
│   ├── ingestion/
│   │   ├── fred_job.py             # FRED data fetcher
│   │   └── sec_job.py              # SEC filing fetcher
│   ├── data/
│   │   ├── clients/                # API wrappers
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/                     # ML pipeline
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│
├── deployment/
│   ├── cloudbuild.yaml             # Automated Docker builds
│   └── docker/
│       └── Dockerfile.airflow      # Airflow container
│
├── tests/                          # Unit + integration tests
├── notebooks/                      # EDA + experiments
├── monitoring/                     # Drift detection
├── Makefile                        # Dev commands
├── pyproject.toml                  # Python dependencies (uv)
└── README.md                       # This file
```

---

## Development

### Local Development

```bash
# Install dependencies
make setup

# Run Airflow locally
make local-up
# Access: http://localhost:8080 (admin/admin)

# Code quality checks
make check          # Format, lint, type check, terraform validate
make format         # Format + fix code
make test           # Run tests

# Run ingestion locally (requires API keys in .env)
source .env
uv run python -m src.ingestion.fred_job
uv run python -m src.ingestion.sec_job
```

### CI/CD

GitHub Actions workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)):
- **checks**: Python formatting, linting, type checking, Terraform validation
- **test**: Unit and integration tests

---

## Configuration

### Required Environment Variables

```bash
# GCP
TF_VAR_project_id=your-gcp-project-id
TF_VAR_region=us-central1

# API Keys
FRED_API_KEY=your-fred-api-key          # Get from https://fred.stlouisfed.org/docs/api/
SEC_USER_AGENT=your-app your@email.com  # Required by SEC API
```

### Getting API Keys

**FRED:**
1. Visit https://fred.stlouisfed.org/docs/api/
2. Click "Request API Key"
3. Add to `.env`: `FRED_API_KEY=...`

**SEC:**
- No key needed
- Provide User-Agent: `SEC_USER_AGENT=app-name user@email.com`

---

## Common Tasks

### Update Infrastructure

```bash
cd infra
terraform plan    # Review changes
terraform apply   # Deploy changes
```

### View Logs

```bash
# Cloud Run logs
gcloud run services logs tail foresight-airflow --region us-central1

# Or use Airflow UI
terraform output airflow_url
```

### Check Data

### Check Data

```bash
# List GCS buckets (replace PROJECT_ID with your project)
gsutil ls gs://PROJECT_ID-foresight-ml-data/

# Check FRED data
gsutil ls gs://PROJECT_ID-foresight-ml-data/raw/fred/

# Check SEC data
gsutil ls gs://PROJECT_ID-foresight-ml-data/raw/sec/

# Query BigQuery
bq query --use_legacy_sql=false "SELECT COUNT(*) FROM \`your-project.foresight_ml.raw_filings\`"
```

### Cleanup

```bash
cd infra
terraform destroy
```

---

## Troubleshooting

### Airflow Not Starting

```bash
# Check Cloud Run logs
gcloud run services logs tail foresight-airflow --region us-central1

# View service status
gcloud run services describe foresight-airflow --region us-central1
```

### Terraform Issues

**Permission Denied Errors:**
```bash
# Re-authenticate with full cloud platform scope
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform

# Enable required APIs
gcloud services enable artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  iam.googleapis.com
```

**Resource Already Exists (e.g., BigQuery dataset):**
```bash
# Import existing resource into Terraform state
cd infra
terraform import google_bigquery_dataset.foresight_ml PROJECT_ID/foresight_ml_dev
terraform plan
```

**Validation:**
```bash
# Validate configuration
make terraform-check

# Or manual validation
cd infra
terraform fmt -check
terraform validate
```

### API Errors

**FRED:**
- Verify API key: https://fred.stlouisfed.org/docs/api/
- Check rate limits: 120 requests/minute

**SEC:**
- Ensure User-Agent header is set correctly
- Rate limit: 10 requests/second

### Permission Errors

```bash
# Check service account roles
gcloud projects get-iam-policy $GCP_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:foresight*"
```

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_data_ingestion.py -v

# Run unit tests only (no API calls)
FRED_API_KEY="" SEC_USER_AGENT="" make test
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | Apache Airflow 2.8+ |
| **Infrastructure** | Terraform, GCP Cloud Run |
| **Storage** | Google Cloud Storage, BigQuery |
| **Language** | Python 3.12 |
| **Package Manager** | uv |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Linting** | Ruff |
| **Type Checking** | mypy |
| **Testing** | pytest |

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [SEC EDGAR API](https://www.sec.gov/edgar/sec-api-documentation)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [uv Package Manager](https://docs.astral.sh/uv/)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests and checks: `make check && make test`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

---

## License

This project is for educational and research purposes.

---

**Last Updated:** February 2026
