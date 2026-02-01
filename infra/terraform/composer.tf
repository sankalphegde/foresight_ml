# Cloud Composer (Managed Airflow)
resource "google_composer_environment" "airflow" {
  count = var.enable_composer ? 1 : 0

  name   = "foresight-ml-${var.environment}"
  region = var.region

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "foresight-ml"
  }

  config {
    software_config {
      image_version = "composer-2-airflow-2.8"

      env_variables = {
        GCS_BUCKET     = google_storage_bucket.data_lake.name
        GCS_CACHE      = google_storage_bucket.cache.name
        ENVIRONMENT    = var.environment
        BQ_DATASET     = google_bigquery_dataset.foresight_ml.dataset_id
      }

      pypi_packages = {
        pandas                   = ">=2.1.4"
        requests                 = ">=2.31.0"
        pydantic                 = ">=2.5.3"
        google-cloud-storage     = ">=2.14.0"
        google-cloud-bigquery    = ">=3.14.0"
      }
    }

    node_config {
      service_account = google_service_account.composer[0].email
    }

    workloads_config {
      scheduler {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        count      = 1
      }

      web_server {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
      }

      worker {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        min_count  = 1
        max_count  = 3
      }
    }

    environment_size = var.composer_environment_size

    # Enable private IP for cost savings (optional)
    # private_environment_config {
    #   enable_private_endpoint = true
    # }
  }
}
