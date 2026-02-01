# Service account for Composer
resource "google_service_account" "composer" {
  count = var.enable_composer ? 1 : 0

  account_id   = "foresight-ml-composer-${var.environment}"
  display_name = "Foresight ML Composer (${var.environment})"
  description  = "Service account for Cloud Composer Airflow environment"
}

# Composer worker role
resource "google_project_iam_member" "composer_worker" {
  count = var.enable_composer ? 1 : 0

  project = var.project_id
  role    = "roles/composer.worker"
  member  = "serviceAccount:${google_service_account.composer[0].email}"
}

# GCS access
resource "google_storage_bucket_iam_member" "composer_gcs_data" {
  count = var.enable_composer ? 1 : 0

  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.composer[0].email}"
}

resource "google_storage_bucket_iam_member" "composer_gcs_cache" {
  count = var.enable_composer ? 1 : 0

  bucket = google_storage_bucket.cache.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.composer[0].email}"
}

# BigQuery access
resource "google_bigquery_dataset_iam_member" "composer_bq" {
  count = var.enable_composer ? 1 : 0

  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.composer[0].email}"
}

# Service account for local development (optional)
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
