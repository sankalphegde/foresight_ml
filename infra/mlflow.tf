# MLflow Tracking Infrastructure (Cloud Run + Cloud SQL + GCS)

resource "google_service_account" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  account_id   = "foresight-mlflow-${var.environment}"
  display_name = "Foresight MLflow (${var.environment})"
  description  = "Service account for MLflow tracking server on Cloud Run"
}

resource "google_project_iam_member" "mlflow_cloudsql_client" {
  count = var.enable_mlflow ? 1 : 0

  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow[0].email}"
}

resource "google_storage_bucket_iam_member" "mlflow_artifacts" {
  count = var.enable_mlflow ? 1 : 0

  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow[0].email}"
}

resource "google_sql_database_instance" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  name                = "foresight-mlflow-${var.environment}"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = false

  settings {
    tier              = var.mlflow_sql_tier
    availability_type = "ZONAL"
    disk_autoresize   = true
    disk_size         = 20

    backup_configuration {
      enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
    }
  }
}

resource "google_sql_database" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  name     = "mlflow"
  instance = google_sql_database_instance.mlflow[0].name
}

resource "google_sql_user" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  instance = google_sql_database_instance.mlflow[0].name
  name     = "mlflow"
  password = var.mlflow_db_password
}

resource "null_resource" "build_mlflow_image" {
  count = var.enable_mlflow ? 1 : 0

  triggers = {
    dockerfile_hash = filemd5("${path.module}/../deployment/docker/Dockerfile.mlflow")
    cloudbuild_hash = filemd5("${path.module}/../deployment/cloudbuild.mlflow.yaml")
    force_rebuild   = "2026-03-15-001"
  }

  provisioner "local-exec" {
    command     = "gcloud builds submit --config=deployment/cloudbuild.mlflow.yaml --substitutions=_REGION=${var.region} ."
    working_dir = "${path.module}/.."
  }

  depends_on = [
    google_artifact_registry_repository.foresight,
  ]
}

resource "google_cloud_run_v2_service" "mlflow" {
  count = var.enable_mlflow ? 1 : 0

  name     = "foresight-mlflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  deletion_protection = false

  template {
    service_account = google_service_account.mlflow[0].email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/foresight/mlflow:latest"

      ports {
        container_port = 5000
      }

      env {
        name  = "BACKEND_STORE_URI"
        value = "postgresql+psycopg2://mlflow:${var.mlflow_db_password}@/${google_sql_database.mlflow[0].name}?host=/cloudsql/${google_sql_database_instance.mlflow[0].connection_name}"
      }

      env {
        name  = "ARTIFACT_ROOT"
        value = "gs://${google_storage_bucket.data_lake.name}/mlflow/artifacts"
      }

      command = ["/bin/sh", "-c"]
      args = [
        "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $${BACKEND_STORE_URI} --artifacts-destination $${ARTIFACT_ROOT} --serve-artifacts",
      ]

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.mlflow[0].connection_name]
      }
    }
  }

  depends_on = [
    null_resource.build_mlflow_image,
    google_project_iam_member.mlflow_cloudsql_client,
    google_storage_bucket_iam_member.mlflow_artifacts,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "mlflow_public_invoker" {
  count = var.enable_mlflow && var.mlflow_allow_unauthenticated ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
