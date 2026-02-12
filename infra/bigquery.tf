# BigQuery dataset
resource "google_bigquery_dataset" "foresight_ml" {
  dataset_id    = "foresight_ml_${var.environment}"
  friendly_name = "Foresight ML ${title(var.environment)}"
  description   = "Financial distress prediction data - ${var.environment}"
  location      = var.region

  default_table_expiration_ms = var.environment == "dev" ? 2592000000 : null # 30 days for dev

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "foresight-ml"
  }
}

# Table for company filings
resource "google_bigquery_table" "company_filings" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  table_id   = "company_filings"

  time_partitioning {
    type  = "DAY"
    field = "filing_date"
  }

  clustering = ["ticker", "form"]

  schema = jsonencode([
    {
      name        = "ticker"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Company ticker symbol"
    },
    {
      name        = "cik"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "SEC Central Index Key"
    },
    {
      name        = "form"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Filing form type (10-K, 10-Q)"
    },
    {
      name        = "filing_date"
      type        = "DATE"
      mode        = "REQUIRED"
      description = "Date the filing was submitted"
    },
    {
      name        = "accession_number"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "SEC accession number"
    },
  ])

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Table for economic indicators
resource "google_bigquery_table" "economic_indicators" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  table_id   = "economic_indicators"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  schema = jsonencode([
    {
      name        = "date"
      type        = "DATE"
      mode        = "REQUIRED"
      description = "Observation date"
    },
    {
      name        = "fed_funds"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "Federal Funds Rate"
    },
    {
      name        = "inflation"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "CPI Inflation Rate"
    },
    {
      name        = "unemployment"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "Unemployment Rate"
    },
    {
      name        = "gdp"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "Gross Domestic Product"
    },
    {
      name        = "credit_spread"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "BBB Credit Spread"
    },
    {
      name        = "vix"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "VIX Volatility Index"
    },
  ])

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Combined view
resource "google_bigquery_table" "company_data" {
  dataset_id = google_bigquery_dataset.foresight_ml.dataset_id
  table_id   = "company_data_view"

  view {
    query = <<-SQL
      SELECT
        f.ticker,
        f.cik,
        f.form,
        f.filing_date,
        f.accession_number,
        e.fed_funds,
        e.inflation,
        e.unemployment,
        e.gdp,
        e.credit_spread,
        e.vix
      FROM `${var.project_id}.${google_bigquery_dataset.foresight_ml.dataset_id}.company_filings` f
      LEFT JOIN `${var.project_id}.${google_bigquery_dataset.foresight_ml.dataset_id}.economic_indicators` e
        ON DATE_TRUNC(f.filing_date, QUARTER) = DATE_TRUNC(e.date, QUARTER)
    SQL

    use_legacy_sql = false
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}
