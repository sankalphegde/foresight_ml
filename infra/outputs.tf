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

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI hosted on Cloud Run"
  value       = var.enable_mlflow ? google_cloud_run_v2_service.mlflow[0].uri : ""
}

output "mlflow_cloudsql_connection_name" {
  description = "Cloud SQL connection name used by MLflow backend"
  value       = var.enable_mlflow ? google_sql_database_instance.mlflow[0].connection_name : ""
}

# Foresight API & Dashboard outputs
output "api_url" {
  description = "Foresight API service URL on Cloud Run"
  value       = google_cloud_run_v2_service.api.uri
}

output "api_service_account_email" {
  description = "API service account email"
  value       = google_service_account.api.email
}

output "dashboard_url" {
  description = "Foresight Dashboard service URL on Cloud Run"
  value       = google_cloud_run_v2_service.dashboard.uri
}

output "dashboard_service_account_email" {
  description = "Dashboard service account email"
  value       = google_service_account.dashboard.email
}

output "slack_alert_adapter_url" {
  description = "Cloud Run URL for the Monitoring-to-Slack webhook adapter"
  value       = local.slack_notifications_enabled ? google_cloud_run_v2_service.slack_alert_adapter[0].uri : ""
  sensitive   = true
}

# output "airflow_url" {
#   description = "Airflow web UI URL (Cloud Run - deprecated)"
#   value       = google_cloud_run_v2_service.airflow.uri
# }

# Instructions
output "setup_instructions" {
  description = "Next steps after deployment"
  value       = <<-EOT
    Deployment complete! Next steps:

    1. Save service account key:
       terraform output -raw dev_service_account_key | base64 -d > gcp-key.json

    2. Configure notification channels:
       - Update alert_emails (or alert_email) in terraform.tfvars
       - Optionally set TF_VAR_slack_webhook_url for Slack alerts

    3. Verify Cloud Run services deployed:
       - API: $(terraform output -raw api_url)
       - Dashboard: $(terraform output -raw dashboard_url)
       - Airflow: $(terraform output -raw airflow_url)

    4. Configure Secret Manager:
       - Set foresight-api-keys secret with actual API keys:
         gcloud secrets versions add foresight-api-keys --data-file=api_keys.json

    5. Monitor via Airflow UI:
       - DAG runs show in the UI
       - Logs available for each task
       - Data stored in GCS: gs://${google_storage_bucket.data_lake.name}

    6. Query data:
       Open BigQuery console: https://console.cloud.google.com/bigquery?project=${var.project_id}

    7. View monitoring alerts, uptime checks, and log metrics:
       https://console.cloud.google.com/monitoring/alerting/policies?project=${var.project_id}

  EOT

}
