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

variable "composer_machine_type" {
  description = "Machine type for Composer worker nodes"
  type        = string
  default     = "n1-standard-1"
}

variable "composer_environment_size" {
  description = "Size of the Composer environment (ENVIRONMENT_SIZE_SMALL, MEDIUM, LARGE)"
  type        = string
  default     = "ENVIRONMENT_SIZE_SMALL"
}
