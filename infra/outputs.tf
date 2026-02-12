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
  description = "Airflow service account email for Cloud Run"
  value       = google_service_account.airflow.email
}

# Cloud Run outputs
output "airflow_url" {
  description = "Airflow web UI URL"
  value       = google_cloud_run_v2_service.airflow.uri
}

# Instructions
output "setup_instructions" {
  description = "Next steps after deployment"
  value       = <<-EOT
    Deployment complete! Next steps:

    1. Save service account key:
       terraform output -raw dev_service_account_key | base64 -d > gcp-key.json

    2. Build and push Airflow image:
       cd deployment
       gcloud builds submit --config cloudbuild.yaml

    3. Access Airflow UI:
       ${google_cloud_run_v2_service.airflow.uri}
       (Login: admin / admin)

    4. The DAG will automatically backfill historical data from 2020-01-01
       With daily runs, all 6 years of data (~2,190 days) will be collected within ~25 days

    5. Monitor via Airflow UI:
       - DAG runs show in the UI
       - Logs available for each task
       - Both FRED and SEC ingestion run in parallel

    6. Query data:
       Open BigQuery console: https://console.cloud.google.com/bigquery?project=${var.project_id}
  EOT
}
