# Service account for Airflow on Cloud Run
resource "google_service_account" "airflow" {
  account_id   = "foresight-ml-airflow-${var.environment}"
  display_name = "Foresight ML Airflow (${var.environment})"
  description  = "Service account for Airflow running on Cloud Run"
}

# GCS access for Airflow
resource "google_storage_bucket_iam_member" "airflow_gcs_data" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.airflow.email}"
}

resource "google_storage_bucket_iam_member" "airflow_gcs_cache" {
  bucket = google_storage_bucket.cache.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.airflow.email}"
}

# BigQuery access for Airflow
resource "google_bigquery_dataset_iam_member" "airflow_bq" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.airflow.email}"
}

# Service account for local development
resource "google_service_account" "dev" {
  account_id   = "foresight-ml-dev-${var.environment}"
  display_name = "Foresight ML Development (${var.environment})"
  description  = "Service account for local development and testing"
}

resource "google_storage_bucket_iam_member" "dev_gcs_data" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dev.email}"
}

resource "google_bigquery_dataset_iam_member" "dev_bq" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.dev.email}"
}

# Service account key for local development
resource "google_service_account_key" "dev_key" {
  service_account_id = google_service_account.dev.name
}
