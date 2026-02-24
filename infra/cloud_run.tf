# ====================================================================
# DEPRECATED: Cloud Run Airflow (replaced by Cloud Composer)
# Keeping this commented for reference. Cloud Composer provides:
# - Persistent Cloud SQL database (vs ephemeral SQLite)
# - Auto-scaling workers
# - Better monitoring and logging
# - Managed upgrades
# ====================================================================

# Uncomment below if you want to use Cloud Run instead of Composer

/*
# Cloud Run Service for Airflow
resource "google_cloud_run_v2_service" "airflow" {
  name     = "foresight-airflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  deletion_protection = false

  template {
    service_account = google_service_account.airflow.email
    timeout         = "300s" # Allow 5 minutes for startup

    scaling {
      min_instance_count = 1 # Keep scheduler running
      max_instance_count = 1 # Single instance for LocalExecutor
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/foresight/airflow:latest"

      ports {
        container_port = 8080
      }

      startup_probe {
        initial_delay_seconds = 60
        timeout_seconds       = 10
        period_seconds        = 15
        failure_threshold     = 20
        tcp_socket {
          port = 8080
        }
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
        name  = "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS"
        value = "admin:admin"
      }
      env {
        name  = "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"
        value = "/home/airflow/simple_auth_manager_passwords.json.generated"
      }
      env {
        name  = "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"
        value = "sqlite:////opt/airflow/data/airflow.db"
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
        mount_path = "/opt/airflow/data"
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

# Require authentication via IAP (users must be granted run.invoker role)
# To grant access: gcloud run services add-iam-policy-binding foresight-airflow \
#   --region=us-central1 \
#   --member="user:YOUR_EMAIL@example.com" \
#   --role="roles/run.invoker"


# Grant access to authorized users
resource "google_cloud_run_v2_service_iam_member" "airflow_users" {
  for_each = toset(var.airflow_authorized_users)

  location = google_cloud_run_v2_service.airflow.location
  name     = google_cloud_run_v2_service.airflow.name
  role     = "roles/run.invoker"
  member   = each.value
}
*/
