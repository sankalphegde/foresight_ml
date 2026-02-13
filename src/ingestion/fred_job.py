"""FRED data ingestion job for Airflow pipeline."""

import os
from datetime import datetime

from google.cloud import storage

from src.data.clients.fred_client import FREDClient


def get_year_month(execution_date: str) -> tuple[int, str]:
    """Extract year and month from execution date string."""
    dt = datetime.fromisoformat(execution_date)
    return dt.year, f"{dt.month:02d}"


def main() -> None:
    """Fetch FRED economic indicators and upload to GCS."""
    # ---- REQUIRED ENV VARS ----
    execution_date = os.environ["EXECUTION_DATE"]
    bucket_name = os.environ["GCS_BUCKET"]
    api_key = os.environ["FRED_API_KEY"]

    year, month = get_year_month(execution_date)

    print("Starting FRED ingestion:", year, month)

    # ---- FETCH DATA ----
    client = FREDClient(api_key=api_key)
    df = client.get_common_indicators(
        start_date="2020-01-01",  # 6 years of economic data for distress patterns
        frequency="q",
    )

    # ---- WRITE TO GCS ----
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    output_path = f"raw/fred/year={year}/month={month}/indicators.csv"

    blob = bucket.blob(output_path)
    blob.upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv",
    )

    print(f"Wrote {len(df)} FRED rows to {output_path}")


if __name__ == "__main__":
    main()
