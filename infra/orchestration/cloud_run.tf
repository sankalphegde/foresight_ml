# Cloud Run Service for Airflow
resource "google_cloud_run_v2_service" "airflow" {
  name     = "foresight-airflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.airflow.email

    scaling {
      min_instance_count = 1 # Keep scheduler running
      max_instance_count = 1 # Single instance for LocalExecutor
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/foresight/airflow:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.data_lake.name
      }
      env {
        name  = "FRED_API_KEY"
        value = var.fred_api_key
      }
      env {
        name  = "SEC_USER_AGENT"
        value = var.sec_user_agent
      }
      env {
        name  = "AIRFLOW__CORE__EXECUTOR"
        value = "LocalExecutor"
      }
      env {
        name  = "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"
        value = "sqlite:////opt/airflow/airflow.db"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      # Persistent disk for SQLite database
      volume_mounts {
        name       = "airflow-db"
        mount_path = "/opt/airflow"
      }
    }

    volumes {
      name = "airflow-db"
      empty_dir {
        medium     = "MEMORY"
        size_limit = "512Mi"
      }
    }
  }

  lifecycle {
    ignore_changes = [
      launch_stage,
    ]
  }

  depends_on = [
    null_resource.build_airflow_image
  ]
}

# Allow public access to Airflow UI (change to IAM for production)
resource "google_cloud_run_v2_service_iam_member" "airflow_public" {
  location = google_cloud_run_v2_service.airflow.location
  name     = google_cloud_run_v2_service.airflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
