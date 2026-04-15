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
# Service account for Foresight API
resource "google_service_account" "api" {
  account_id   = "foresight-api-${var.environment}"
  display_name = "Foresight API (${var.environment})"
  description  = "Service account for Foresight API running on Cloud Run"
}

# Grant API service account read access to GCS
resource "google_storage_bucket_iam_member" "api_gcs_read" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# Grant API service account access to Secret Manager
resource "google_project_iam_member" "api_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Service account for Foresight Dashboard
resource "google_service_account" "dashboard" {
  account_id   = "foresight-dashboard-${var.environment}"
  display_name = "Foresight Dashboard (${var.environment})"
  description  = "Service account for Foresight Dashboard running on Cloud Run"
}

# Grant Dashboard service account read access to GCS
resource "google_storage_bucket_iam_member" "dashboard_gcs_read" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dashboard.email}"
}

# Grant Dashboard service account read access to the legacy populated bucket
resource "google_storage_bucket_iam_member" "dashboard_gcs_read_legacy" {
  bucket = "financial-distress-data"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dashboard.email}"
}

# Grant API service account read access to the legacy populated bucket
resource "google_storage_bucket_iam_member" "api_gcs_read_legacy" {
  bucket = "financial-distress-data"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}
