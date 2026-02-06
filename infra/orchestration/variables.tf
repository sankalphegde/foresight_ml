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
