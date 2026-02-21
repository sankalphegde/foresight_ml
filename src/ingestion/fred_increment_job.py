"""Production FRED ingestion.

Stores full time series per series_id.
"""

import os
import pandas as pd
from google.cloud import storage
from src.data.clients.fred_client import FREDClient

BUCKET = os.environ["GCS_BUCKET"]
API_KEY = os.environ["FRED_API_KEY"]
RAW_PREFIX = "raw/fred"


def load_existing(storage_client: "storage.Client", series_id: str) -> "pd.DataFrame | None":
    """Load existing FRED data from cloud storage."""
    blob_path = f"{RAW_PREFIX}/series_id={series_id}.parquet"
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    with blob.open("rb") as f:
        return pd.read_parquet(f)  # type: ignore


def save(storage_client: "storage.Client", series_id: str, df: "pd.DataFrame") -> None:
    """Save updated FRED data to cloud storage."""
    blob_path = f"{RAW_PREFIX}/series_id={series_id}.parquet"
    bucket = storage_client.bucket(BUCKET)
    with bucket.blob(blob_path).open("wb") as f:
        df.to_parquet(f, index=False)  # type: ignore
    print("Saved", blob_path)


def main() -> None: 
    """Execute the main ingestion pipeline for FRED data."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    storage_client = storage.Client(project=project_id)
    fred = FREDClient(api_key=API_KEY)

    for _name, series_id in fred.INDICATORS.items():
        print("Fetching", series_id)

        # Fetch native frequency
        new_df = fred.get_series_dataframe(series_id)
        new_df = new_df.reset_index()

        new_df["date"] = pd.to_datetime(new_df["date"])

        # Convert to quarterly
        new_df["quarter_end_date"] = (
            new_df["date"]
            .dt.to_period("Q")
            .dt.to_timestamp("Q")
        )

        new_df = (
            new_df
            .sort_values("date")
            .groupby("quarter_end_date")["value"]
            .last()
            .reset_index()
        )

        new_df = new_df.rename(columns={"quarter_end_date": "date"})

        # Load existing
        existing_df = load_existing(storage_client, series_id)

        if existing_df is None:
            print("Full historical load.")
            save(storage_client, series_id, new_df)
            continue

        # Revision-safe refresh
        REFRESH_LAST_N = 8

        existing_df["date"] = pd.to_datetime(existing_df["date"])

        latest_dates = (
            existing_df["date"]
            .drop_duplicates()
            .sort_values()
            .tail(REFRESH_LAST_N)
        )

        existing_df = existing_df[
            ~existing_df["date"].isin(latest_dates)
        ]

        old_dates = set(existing_df["date"].unique())

        incremental_df = new_df[
            ~new_df["date"].isin(old_dates)
        ]

        updated = pd.concat([existing_df, incremental_df], ignore_index=True)

        updated = updated.drop_duplicates(
            subset=["date"],
            keep="last"
        ).sort_values("date")

        save(storage_client, series_id, updated)

if __name__ == "__main__":
    main()