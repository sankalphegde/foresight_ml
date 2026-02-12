# Setup Guide

## Quick Deploy

### 1. Configure Environment

**Option A: Using environment variables (recommended)**

```bash
# Copy and edit environment file
cp example.env .env
# Edit .env with your values (project_id, API keys, etc.)

# Load environment variables
source .env

# Verify
echo $TF_VAR_project_id
echo $FRED_API_KEY
```

**Option B: Using terraform.tfvars file**

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

> **Note:** With environment variables, Terraform automatically reads `TF_VAR_*` variables. No terraform.tfvars file needed!

### 2. Deploy Infrastructure

```bash
cd infra
terraform init
terraform plan
terraform apply
```

Creates: GCS buckets, BigQuery dataset, Artifact Registry, builds Docker image via Cloud Build, deploys Cloud Run service, service accounts, uploads companies.csv

### 3. Access Airflow

```bash
terraform output airflow_url
```

Login: `admin` / `admin`
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
      │  GCS Bucket │
      │  BigQuery   │
      └─────────────┘
```

**Data Collection:**
- Historical: 2020-01-01 to present (6 years)
- Schedule: Daily with catchup enabled
- Timeline: ~25 days to complete backfill
- Concurrent runs: 3 max

---

## Local Development

### Prerequisites

**Install Terraform:**

*macOS:*
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

*Linux:*
```bash
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

*Windows:*
```powershell
choco install terraform
```

Or download from https://developer.hashicorp.com/terraform/downloads

**Install gcloud CLI (Optional):**

*macOS:*
```bash
brew install google-cloud-sdk
```

*Linux:*
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

*Windows:*
Download from https://cloud.google.com/sdk/docs/install

**Authenticate:**
```bash
gcloud auth login
gcloud config set project $GCP_PROJECT_ID
```

### First-time Setup

```bash
# Install dependencies
make setup

# Configure environment variables
cp example.env .env
# Edit .env with your values

# Load environment
source .env

# Create GCP service account key (for local development)
# If service account already exists, skip the create step
gcloud iam service-accounts create foresight-dev \
  --display-name="Foresight ML Local Dev" \
  2>/dev/null || echo "Service account already exists, continuing..."

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:foresight-dev@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

mkdir -p .gcp
gcloud iam service-accounts keys create .gcp/service-account-key.json \
  --iam-account="foresight-dev@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
```

### Run Airflow Locally

```bash
# Start Airflow
docker-compose up airflow

# Access UI: http://localhost:8080 (admin/admin)
```

### Code Quality

```bash
# Run all checks (formatting, linting, type checking, terraform validation)
make check

# Run individual checks
make format          # Format and fix code
make typecheck       # Type check with mypy
make terraform-check # Validate Terraform configuration

# Run unit tests only (no API calls)
FRED_API_KEY="" SEC_USER_AGENT="" make test 2>&1 | grep -E "(PASSED|test_.*_client)"

# Run all tests including integration (requires API keys in .env)
source .env
make test
```
---

## GitHub Actions Setup

To enable integration tests in CI/CD, configure GitHub secrets:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

| Secret Name | Value | Example |
|------------|-------|---------|
| `FRED_API_KEY` | Your FRED API key | `abcd1234...` |
| `SEC_USER_AGENT` | Your user agent with email | `foresight-ml your@example.com` |

---

## Maintenance

**Update Infrastructure or Code:**
```bash
cd infra
terraform plan
terraform apply

# Terraform will automatically rebuild Docker image if Dockerfile changed
# To force rebuild without changes, edit force_rebuild in artifact_registry.tf
```

**View Logs:**
```bash
gcloud run services logs tail foresight-airflow --region us-central1
# Or view in Airflow UI
```

**Cleanup:**
```bash
cd infra
terraform destroy
```
