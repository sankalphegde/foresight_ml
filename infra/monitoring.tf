locals {
  alert_email_recipients      = distinct(compact(concat(var.alert_emails, var.alert_email != "" ? [var.alert_email] : [])))
  slack_notifications_enabled = var.slack_webhook_url != ""
  alert_notification_channel_ids = concat(
    [for ch in google_monitoring_notification_channel.email : ch.id],
    local.slack_notifications_enabled ? [google_monitoring_notification_channel.slack[0].id] : []
  )
  api_uptime_host = trimprefix(trimsuffix(google_cloud_run_v2_service.api.uri, "/"), "https://")
}

resource "google_monitoring_metric_descriptor" "model_roc_auc" {
  type         = "custom.googleapis.com/model/roc_auc"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  display_name = "Model ROC-AUC"

  labels {
    key         = "project_id"
    value_type  = "STRING"
    description = "Project identifier"
  }
}

resource "google_monitoring_metric_descriptor" "data_drift_score" {
  type         = "custom.googleapis.com/data/drift_score"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  display_name = "Data Drift Score"

  labels {
    key         = "project_id"
    value_type  = "STRING"
    description = "Project identifier"
  }
}

# Log-based metric for Cloud Run job failures.
resource "google_logging_metric" "cloud_run_job_failures" {
  name        = "cloud_run_job_failures"
  description = "Count of Cloud Run job error logs"

  filter = <<-EOT
    resource.type="cloud_run_job"
    severity>=ERROR
  EOT
}

# Log-based metric for low ROC-AUC runs in the training job.
resource "google_logging_metric" "test_roc_auc_low" {
  name        = "test_roc_auc_low"
  description = "Count of training logs with ROC-AUC below the 0.85 alert threshold"

  filter = <<-EOT
    resource.type="cloud_run_job"
    textPayload=~"TEST_ROC_AUC_LOW"
  EOT
}

# Log-based metric for drift alerts emitted by the feature engineering pipeline.
resource "google_logging_metric" "drift_detected" {
  name        = "drift_detected"
  description = "Count of drift alert logs emitted by the feature engineering pipeline"

  filter = <<-EOT
    resource.type="cloud_run_job"
    textPayload=~"DRIFT_DETECTED"
  EOT
}

# Notification channels for email alerts
resource "google_monitoring_notification_channel" "email" {
  for_each     = toset(local.alert_email_recipients)
  display_name = "Foresight Email Notifications (${var.environment}) - ${each.value}"
  type         = "email"
  enabled      = true

  labels = {
    email_address = each.value
  }
}

# Optional Slack notification channel via Incoming Webhook URL
resource "google_monitoring_notification_channel" "slack" {
  count        = local.slack_notifications_enabled ? 1 : 0
  display_name = "Foresight Slack Webhook Notifications (${var.environment})"
  type         = "webhook_basicauth"
  enabled      = true

  labels = {
    url      = google_cloud_run_v2_service.slack_alert_adapter[0].uri
    username = "foresight"
    password = "unused"
  }

  depends_on = [
    google_cloud_run_v2_service.slack_alert_adapter,
    google_cloud_run_v2_service_iam_member.slack_alert_adapter_public,
  ]
}

# Alert Policy 1: Cloud Run Job Failure
resource "google_monitoring_alert_policy" "cloud_run_job_failure" {
  display_name = "Cloud Run Job Failure (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Cloud Run job logs include ERROR"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_job\" AND metric.type=\"logging.googleapis.com/user/cloud_run_job_failures\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }

  notification_channels = local.alert_notification_channel_ids

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "A Cloud Run job emitted an ERROR log. Check the job execution logs in Cloud Run and the linked job history."
    mime_type = "text/markdown"
  }

  depends_on = [
    google_logging_metric.cloud_run_job_failures,
    google_monitoring_notification_channel.email,
    google_monitoring_notification_channel.slack,
  ]
}

# Alert Policy 2: Model ROC-AUC below 0.85 (email only)
resource "google_monitoring_alert_policy" "test_roc_auc_low" {
  display_name = "Model Test ROC-AUC Below 0.85 (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Training logs include TEST_ROC_AUC_LOW"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_job\" AND metric.type=\"logging.googleapis.com/user/test_roc_auc_low\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }

  notification_channels = [for ch in google_monitoring_notification_channel.email : ch.id]

  alert_strategy {
    auto_close = "3600s"
  }

  documentation {
    content   = "Training produced a ROC-AUC below 0.85. Review the latest Cloud Run job output and MLflow run metrics."
    mime_type = "text/markdown"
  }

  depends_on = [
    google_logging_metric.test_roc_auc_low,
    google_monitoring_notification_channel.email,
  ]
}

# Alert Policy 3: Drift detected (Slack only)
resource "google_monitoring_alert_policy" "drift_detected" {
  display_name = "Drift Detected (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Feature engineering logs include DRIFT_DETECTED"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_job\" AND metric.type=\"logging.googleapis.com/user/drift_detected\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }

  notification_channels = local.slack_notifications_enabled ? [google_monitoring_notification_channel.slack[0].id] : []

  alert_strategy {
    auto_close = "7200s"
  }

  documentation {
    content   = "Drift was detected in the feature engineering pipeline. Review the bias report and recent pipeline execution logs."
    mime_type = "text/markdown"
  }

  depends_on = [
    google_logging_metric.drift_detected,
    google_monitoring_notification_channel.slack,
  ]
}

# Uptime check for the public API health endpoint
resource "google_monitoring_uptime_check_config" "api_health" {
  display_name = "Foresight API Health Uptime (${var.environment})"
  timeout      = "10s"
  period       = "60s"

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = local.api_uptime_host
    }
  }

  http_check {
    path           = "/health"
    port           = 443
    use_ssl        = true
    validate_ssl   = true
    request_method = "GET"
  }

  selected_regions = ["USA", "EUROPE", "ASIA_PACIFIC"]
}

# Alert Policy 6: API uptime check failures
resource "google_monitoring_alert_policy" "api_uptime_check_failed" {
  display_name = "Foresight API Uptime Check Failed (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Uptime check pass ratio < 1"

    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.labels.check_id=\"${google_monitoring_uptime_check_config.api_health.uptime_check_id}\" AND resource.type=\"uptime_url\""
      duration        = "120s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period   = "120s"
        per_series_aligner = "ALIGN_NEXT_OLDER"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = local.alert_notification_channel_ids

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "The API uptime check for /health is failing. Verify Cloud Run service health and recent deployments."
    mime_type = "text/markdown"
  }

  depends_on = [
    google_monitoring_uptime_check_config.api_health,
    google_monitoring_notification_channel.email,
    google_monitoring_notification_channel.slack,
  ]
}
