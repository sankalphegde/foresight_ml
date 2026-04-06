resource "null_resource" "build_slack_adapter_image" {
  triggers = {
    dockerfile_hash = filemd5("${path.module}/../deployment/docker/Dockerfile.slack_adapter")
    source_hash     = filemd5("${path.module}/../deployment/slack_adapter/app.py")
    requirements    = filemd5("${path.module}/../deployment/slack_adapter/requirements.txt")
    cloudbuild_hash = filemd5("${path.module}/../deployment/cloudbuild.slack_adapter.yaml")
    force_rebuild   = "2026-04-06-001"
  }

  provisioner "local-exec" {
    command     = "gcloud builds submit --config=deployment/cloudbuild.slack_adapter.yaml --substitutions=_REGION=${var.region} ."
    working_dir = "${path.module}/.."
  }

  depends_on = [google_artifact_registry_repository.foresight]
}

resource "google_cloud_run_v2_service" "slack_alert_adapter" {
  count = local.slack_notifications_enabled ? 1 : 0

  name                = "foresight-slack-alert-adapter"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    timeout = "30s"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/foresight/slack-alert-adapter:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "SLACK_WEBHOOK_URL"
        value = var.slack_webhook_url
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [launch_stage]
  }

  depends_on = [null_resource.build_slack_adapter_image]
}

resource "google_cloud_run_v2_service_iam_member" "slack_alert_adapter_public" {
  count = local.slack_notifications_enabled ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.slack_alert_adapter[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
