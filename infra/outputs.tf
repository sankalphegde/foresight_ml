output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

# Storage outputs
output "data_lake_bucket" {
  description = "GCS bucket for data lake"
  value       = google_storage_bucket.data_lake.name
}

output "cache_bucket" {
  description = "GCS bucket for cache"
  value       = google_storage_bucket.cache.name
}

output "data_lake_url" {
  description = "Console URL for data lake bucket"
  value       = "https://console.cloud.google.com/storage/browser/${google_storage_bucket.data_lake.name}"
}

# BigQuery outputs
output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.foresight_ml.dataset_id
}

output "bigquery_console_url" {
  description = "Console URL for BigQuery dataset"
  value       = "https://console.cloud.google.com/bigquery?project=${var.project_id}&d=${google_bigquery_dataset.foresight_ml.dataset_id}"
}

output "bigquery_view" {
  description = "BigQuery view for combined company data"
  value       = "${var.project_id}.${google_bigquery_dataset.foresight_ml.dataset_id}.${google_bigquery_table.company_data.table_id}"
}

# Service account outputs
output "dev_service_account_email" {
  description = "Development service account email"
  value       = google_service_account.dev.email
}

output "dev_service_account_key" {
  description = "Development service account key (base64 encoded)"
  value       = google_service_account_key.dev_key.private_key
  sensitive   = true
}

output "airflow_service_account_email" {
  description = "Airflow service account email for Cloud Run (deprecated)"
  value       = google_service_account.airflow.email
}

output "composer_service_account_email" {
  description = "Composer service account email"
  value       = google_service_account.composer.email
}

# Cloud Composer outputs
output "composer_airflow_uri" {
  description = "Cloud Composer Airflow web UI URL"
  value       = google_composer_environment.foresight_ml.config[0].airflow_uri
}

output "composer_dag_gcs_prefix" {
  description = "GCS path where Composer expects DAG files"
  value       = google_composer_environment.foresight_ml.config[0].dag_gcs_prefix
}

output "composer_environment_name" {
  description = "Cloud Composer environment name"
  value       = google_composer_environment.foresight_ml.name
}

# Cloud Run outputs (deprecated - keeping for reference)
# output "airflow_url" {
#   description = "Airflow web UI URL (Cloud Run - deprecated)"
#   value       = google_cloud_run_v2_service.airflow.uri
# }

# Instructions
output "setup_instructions" {
  description = "Next steps after deployment"
  value       = <<-EOT
    Deployment complete! Next steps:

    1. Save service account key (for local development):
       terraform output -raw dev_service_account_key | base64 -d > gcp-key.json

    2. Upload DAGs to Cloud Composer:
       gcloud composer environments storage dags import \
         --environment=foresight-ml-${var.environment} \
         --location=${var.region} \
         --source=src/airflow/dags/

    3. Access Airflow UI:
       Open: $(terraform output -raw composer_airflow_uri)
       Authentication: Use your Google Cloud credentials

    4. Available DAGs:
       - foresight_ingestion: Daily data ingestion (FRED + SEC)
       - foresight_preprocessing: Weekly data preprocessing

    5. Monitor via Airflow UI:
       - DAG runs show in the UI
       - Logs available for each task
       - Data stored in GCS: gs://${google_storage_bucket.data_lake.name}

    6. Query processed data:
       Open BigQuery console: https://console.cloud.google.com/bigquery?project=${var.project_id}

    Note: Cloud Composer takes 20-30 minutes to provision on first run.
  EOT
}
