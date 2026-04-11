# Cloud Run Service for Foresight Dashboard
resource "google_cloud_run_v2_service" "dashboard" {
  name     = "foresight-dashboard"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  deletion_protection = false

  template {
    service_account = google_service_account.dashboard.email
    timeout         = "3600s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.dashboard_image

      ports {
        container_port = 8080
      }

      startup_probe {
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 10
        http_get {
          path = "/health"
          port = 8080
        }
      }

      liveness_probe {
        initial_delay_seconds = 60
        timeout_seconds       = 5
        period_seconds        = 30
        http_get {
          path = "/health"
          port = 8080
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      env {
        name  = "FORESIGHT_API_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      env {
        name  = "GCS_BUCKET"
        value = "financial-distress-data"
      }

      env {
        name  = "MLFLOW_TRACKING_URI"
        value = var.enable_mlflow ? google_cloud_run_v2_service.mlflow[0].uri : ""
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "LOG_LEVEL"
        value = var.environment == "prod" ? "INFO" : "DEBUG"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      launch_stage,
    ]
  }

  depends_on = [
    google_service_account.dashboard,
    google_cloud_run_v2_service.api,
  ]
}

# Allow public access to Dashboard
resource "google_cloud_run_v2_service_iam_member" "dashboard_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.dashboard.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
