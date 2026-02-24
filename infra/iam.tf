# Service account for Airflow on Cloud Run
resource "google_service_account" "airflow" {
  account_id   = "foresight-ml-airflow-${var.environment}"
  display_name = "Foresight ML Airflow (${var.environment})"
  description  = "Service account for Airflow running on Cloud Run"
}

# Service account for Cloud Composer
resource "google_service_account" "composer" {
  account_id   = "foresight-ml-composer-${var.environment}"
  display_name = "Foresight ML Composer (${var.environment})"
  description  = "Service account for Cloud Composer managed Airflow"
}

# Grant Composer permissions to the service account
resource "google_project_iam_member" "composer_worker" {
  project = var.project_id
  role    = "roles/composer.worker"
  member  = "serviceAccount:${google_service_account.composer.email}"
}

# GCS access for Composer
resource "google_storage_bucket_iam_member" "composer_gcs_data" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.composer.email}"
}

resource "google_storage_bucket_iam_member" "composer_gcs_cache" {
  bucket = google_storage_bucket.cache.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.composer.email}"
}

# BigQuery access for Composer
resource "google_bigquery_dataset_iam_member" "composer_bq" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.composer.email}"
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
