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

# Composer outputs
output "composer_airflow_uri" {
  description = "Airflow webserver URL"
  value       = var.enable_composer ? google_composer_environment.airflow[0].config[0].airflow_uri : "Composer disabled"
}

output "composer_dag_gcs_prefix" {
  description = "GCS path for uploading DAGs"
  value       = var.enable_composer ? google_composer_environment.airflow[0].config[0].dag_gcs_prefix : "Composer disabled"
}

output "composer_environment_name" {
  description = "Cloud Composer environment name"
  value       = var.enable_composer ? google_composer_environment.airflow[0].name : "Composer disabled"
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

# Instructions
output "setup_instructions" {
  description = "Next steps after deployment"
  value = <<-EOT
    Deployment complete! Next steps:

    1. Save service account key:
       terraform output -raw dev_service_account_key | base64 -d > gcp-key.json

    2. Set environment variables:
       export GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
       export GCS_BUCKET=${google_storage_bucket.data_lake.name}

    3. Access Airflow UI:
       ${var.enable_composer ? google_composer_environment.airflow[0].config[0].airflow_uri : "Composer disabled - use docker-compose for local"}

    4. Upload DAGs (if Composer enabled):
       gcloud composer environments storage dags import \
         --environment ${var.enable_composer ? google_composer_environment.airflow[0].name : "N/A"} \
         --location ${var.region} \
         --source airflow/dags/data_pipeline.py

    5. Query data:
       Open BigQuery console: https://console.cloud.google.com/bigquery?project=${var.project_id}
  EOT
}
