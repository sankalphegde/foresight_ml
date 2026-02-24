# Cloud Composer (Managed Airflow) Environment
# This replaces the ephemeral Cloud Run deployment with a persistent,
# fully-managed Airflow instance with Cloud SQL backend.
#
# Required GCP APIs (enable before terraform apply):
#   - composer.googleapis.com
#   - cloudsql.googleapis.com
#   - compute.googleapis.com
#   - container.googleapis.com
#
# Enable with:
#   gcloud services enable composer.googleapis.com cloudsql.googleapis.com \
#     compute.googleapis.com container.googleapis.com
#
# Provisioning time: 20-30 minutes on first creation

resource "google_composer_environment" "foresight_ml" {
  name    = "foresight-ml-${var.environment}"
  region  = var.region
  project = var.project_id

  config {
    # Software configuration
    software_config {
      airflow_config_overrides = {
        "core-dags_are_paused_at_creation" = "True"
        "core-max_active_runs_per_dag"     = "3"
        "webserver-expose_config"          = "True"
        "scheduler-catchup_by_default"     = "False"
      }

      env_variables = {
        GCP_PROJECT_ID = var.project_id
        GCS_BUCKET     = google_storage_bucket.data_lake.name
        FRED_API_KEY   = var.fred_api_key
        SEC_USER_AGENT = var.sec_user_agent
        ENVIRONMENT    = var.environment
      }

      # Composer 2 - runs Airflow 2.x
      image_version = "composer-2-airflow-2"

      pypi_packages = {
        # Add Python packages needed by your DAGs
        pandas                = ">=2.0.0"
        requests              = ">=2.31.0"
        pydantic              = ">=2.0.0"
        pyarrow               = ">=14.0.0"
        google-cloud-storage  = ">=2.10.0"
        google-cloud-bigquery = ">=3.11.0"
      }
    }

    # Node configuration for workers
    node_config {
      service_account = google_service_account.composer.email

      disk_size_gb = 30
      machine_type = var.composer_machine_type

      # Network configuration
      network    = "default"
      subnetwork = "default"

      tags = ["composer-worker", "foresight-ml"]
    }

    # Workloads configuration (Composer 2)
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

    # Environment size
    environment_size = var.composer_environment_size

    # Enable private IP (recommended for production)
    # private_environment_config {
    #   enable_private_endpoint = false
    # }
  }

  depends_on = [
    google_service_account.composer,
    google_project_iam_member.composer_worker,
  ]

  lifecycle {
    # Prevent accidental deletion
    prevent_destroy = false
  }
}
