# Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "foresight" {
  location      = var.region
  repository_id = "foresight"
  description   = "Docker repository for Foresight ML images"
  format        = "DOCKER"
}

# Trigger Cloud Build to build and push Airflow image
resource "null_resource" "build_airflow_image" {
  # Trigger rebuild when Dockerfile or source code changes
  triggers = {
    dockerfile_hash = filemd5("${path.module}/../../deployment/docker/Dockerfile.airflow")
    cloudbuild_hash = filemd5("${path.module}/../../deployment/cloudbuild.yaml")
    # Update this timestamp to force a rebuild: 2026-02-06
    force_rebuild = "2026-02-06-001"
  }

  provisioner "local-exec" {
    command     = "gcloud builds submit --config cloudbuild.yaml"
    working_dir = "${path.module}/../../deployment"
  }

  depends_on = [
    google_artifact_registry_repository.foresight
  ]
}
