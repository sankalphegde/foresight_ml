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

variable "composer_environment_size" {
  description = "Cloud Composer environment size"
  type        = string
  default     = "ENVIRONMENT_SIZE_SMALL"
  validation {
    condition = contains([
      "ENVIRONMENT_SIZE_SMALL",
      "ENVIRONMENT_SIZE_MEDIUM",
      "ENVIRONMENT_SIZE_LARGE"
    ], var.composer_environment_size)
    error_message = "Must be SMALL, MEDIUM, or LARGE"
  }
}

variable "data_retention_days" {
  description = "Number of days to retain data in standard storage"
  type        = number
  default     = 90
}

variable "enable_composer" {
  description = "Whether to deploy Cloud Composer (expensive)"
  type        = bool
  default     = true
}
