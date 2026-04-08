variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "data_retention_days" {
  description = "Number of days to retain data in standard storage"
  type        = number
  default     = 90
}

variable "fred_api_key" {
  description = "FRED API key for economic data"
  type        = string
  sensitive   = true
}

variable "sec_user_agent" {
  description = "User agent for SEC API requests"
  type        = string
  default     = "foresight-ml contact@example.com"
}

variable "airflow_authorized_users" {
  description = "List of users authorized to access Airflow (e.g., user:email@example.com)"
  type        = list(string)
  default     = []
}

variable "enable_mlflow" {
  description = "Enable MLflow tracking infrastructure (Cloud Run + Cloud SQL)"
  type        = bool
  default     = true
}

variable "mlflow_sql_tier" {
  description = "Cloud SQL machine tier for MLflow metadata backend"
  type        = string
  default     = "db-custom-1-3840"
}

variable "mlflow_db_password" {
  description = "Password for MLflow PostgreSQL user"
  type        = string
  sensitive   = true
}

variable "mlflow_allow_unauthenticated" {
  description = "Allow unauthenticated access to MLflow Cloud Run service"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Single email address for alert notifications (deprecated; use alert_emails)"
  type        = string
  default     = ""
}

variable "alert_emails" {
  description = "List of email addresses for alert notifications"
  type        = list(string)
  default     = []
}

variable "slack_webhook_url" {
  description = "Slack Incoming Webhook URL for Monitoring alert notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "create_initial_api_keys_version" {
  description = "Whether to create an initial version for the foresight-api-keys secret"
  type        = bool
  default     = false
}

variable "api_keys_secret_payload" {
  description = "API keys payload stored in Secret Manager as JSON (e.g., {fred_api_key = \"...\"})"
  type        = map(string)
  sensitive   = true
  default     = {}
}

variable "api_image" {
  description = "Container image for the API Cloud Run service"
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "dashboard_image" {
  description = "Container image for the dashboard Cloud Run service"
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}
