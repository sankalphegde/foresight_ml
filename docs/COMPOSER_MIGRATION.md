# Migration Guide: Cloud Run → Cloud Composer

## Overview

We've migrated from **Cloud Run** (ephemeral Airflow) to **Cloud Composer** (managed Airflow) for better reliability and persistence.

### Why the Change?

| Issue with Cloud Run | Cloud Composer Solution |
|---------------------|------------------------|
| Ephemeral SQLite database (resets on restart) | Persistent Cloud SQL database |
| Manual password management | Google IAM authentication |
| No auto-scaling for workers | Auto-scaling worker pools |
| Limited monitoring | Full GCP monitoring integration |
| Container restarts lose state | State persists across updates |

---

## Prerequisites

### 1. Enable Required APIs

```bash
gcloud services enable \
  composer.googleapis.com \
  cloudsql.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com
```

### 2. Set Environment Variables

```bash
source .env  # Load your .env file with TF_VAR_* variables
```

---

## Deployment Steps

### 1. Deploy Infrastructure

```bash
cd infra
terraform init
terraform plan   # Review changes (Cloud Composer creation, Cloud Run deletion)
terraform apply
```

**⏱️ Expected time:** 20-30 minutes for first Composer environment creation

### 2. Verify Deployment

```bash
# Check Composer status
make composer-status

# Or directly:
gcloud composer environments list --locations=us-central1
```

### 3. Deploy DAGs

```bash
# Deploy all DAGs to Composer
make composer-deploy

# Or manually:
gcloud composer environments storage dags import \
  --environment=foresight-ml-dev \
  --location=us-central1 \
  --source=src/airflow/dags/
```

### 4. Access Airflow UI

```bash
# Get the Airflow UI URL
terraform output composer_airflow_uri

# Or:
gcloud composer environments describe foresight-ml-dev \
  --location=us-central1 \
  --format="get(config.airflowUri)"
```

Authentication: Use your **Google Cloud credentials** (no more manual passwords!)

---

## Available DAGs

### 1. `foresight_ingestion`
**Schedule:** Daily
**Purpose:** Ingest FRED economic indicators and SEC XBRL filings
**Tasks:**
- `run_fred_ingestion`: Fetch FRED data
- `run_sec_ingestion`: Fetch SEC filings

### 2. `foresight_preprocessing` (NEW!)
**Schedule:** Weekly
**Purpose:** Preprocess raw data into clean interim format
**Tasks:**
1. `prepare_directories`: Create temp directories
2. `sync_sec_raw_data`: Download SEC data from GCS
3. `sync_fred_raw_data`: Download FRED data from GCS
4. `run_preprocessing`: Clean, format, deduplicate
5. `upload_sec_interim`: Upload processed SEC data
6. `upload_fred_interim`: Upload processed FRED data
7. `upload_report`: Upload validation report
8. `cleanup_temp_files`: Clean up temp files

---

## Key Differences

### Authentication
- **Before (Cloud Run):** Username/password (ephemeral, auto-generated)
- **After (Composer):** Google IAM (use your GCP credentials)

### Database
- **Before:** SQLite in ephemeral container storage
- **After:** Cloud SQL (persistent PostgreSQL)

### Scaling
- **Before:** Single container, fixed resources
- **After:** Auto-scaling workers (1-3 workers by default)

### Monitoring
- **Before:** Cloud Run logs only
- **After:** Full Airflow UI + GCP monitoring + Cloud Logging

### Cost
- **Before:** ~$15-30/month (always-on Cloud Run)
- **After:** ~$250-350/month (Cloud Composer small environment)
  - Includes: Cloud SQL, GKE nodes, managed services

---

## Updating DAGs

### Method 1: Using Makefile (Recommended)

```bash
# Edit your DAG files in src/airflow/dags/
vim src/airflow/dags/foresight_ml_preprocessing_pipeline.py

# Deploy changes
make composer-deploy
```

### Method 2: Using gcloud

```bash
gcloud composer environments storage dags import \
  --environment=foresight-ml-dev \
  --location=us-central1 \
  --source=src/airflow/dags/foresight_ml_preprocessing_pipeline.py
```

### Method 3: Direct GCS Upload

```bash
# Get the DAG bucket
DAG_BUCKET=$(terraform output -raw composer_dag_gcs_prefix)

# Upload directly
gsutil cp src/airflow/dags/*.py $DAG_BUCKET/
```

DAGs are automatically picked up within 1-2 minutes.

---

## Python Package Management

### Adding New Dependencies

Edit `infra/composer.tf`:

```hcl
pypi_packages = {
  pandas      = ">=2.0.0"
  your_package = ">=1.0.0"  # Add here
}
```

Then apply:

```bash
cd infra
terraform apply
```

**⏱️ Expected time:** 5-10 minutes to update packages

---

## Troubleshooting

### DAGs Not Appearing

1. Check DAG is uploaded:
   ```bash
   gsutil ls $(terraform output -raw composer_dag_gcs_prefix)
   ```

2. Check for syntax errors in Airflow UI → Admin → Import Errors

3. Force refresh (takes 1-2 minutes):
   ```bash
   gcloud composer environments run foresight-ml-dev \
     --location=us-central1 dags list-import-errors
   ```

### Environment Taking Too Long

First-time provisioning can take 20-30 minutes. Check status:

```bash
gcloud composer environments describe foresight-ml-dev \
  --location=us-central1 \
  --format="get(state)"
```

States: `CREATING` → `RUNNING` → `READY`

### Task Failures

1. View logs in Airflow UI → Task → Logs
2. Or check Cloud Logging:
   ```bash
   gcloud logging read "resource.type=cloud_composer_environment" --limit=50
   ```

### Access Denied to Airflow UI

Grant yourself Composer User role:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL@example.com" \
  --role="roles/composer.user"
```

---

## Rollback to Cloud Run (If Needed)

If you need to temporarily rollback:

1. Uncomment Cloud Run resources in `infra/cloud_run.tf` and `infra/artifact_registry.tf`
2. Comment out Composer resources in `infra/composer.tf`
3. Apply:
   ```bash
   cd infra
   terraform apply
   ```

**Note:** Not recommended long-term due to Cloud Run's limitations.

---

## Cost Optimization

### Development Environment

Use the smallest environment size:

```hcl
# In infra/terraform.tfvars or .env
TF_VAR_composer_environment_size="ENVIRONMENT_SIZE_SMALL"
TF_VAR_composer_machine_type="n1-standard-1"
```

### Production Environment

Scale up for better performance:

```hcl
TF_VAR_composer_environment_size="ENVIRONMENT_SIZE_MEDIUM"
TF_VAR_composer_machine_type="n1-standard-2"
```

---

## Next Steps

1. ✅ Deploy Cloud Composer with `terraform apply`
2. ✅ Deploy DAGs with `make composer-deploy`
3. ✅ Access Airflow UI and enable DAGs
4. ✅ Monitor first runs in Airflow UI
5. ✅ Set up alerts in GCP Monitoring (optional)

---

## Additional Resources

- [Cloud Composer Documentation](https://cloud.google.com/composer/docs)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
