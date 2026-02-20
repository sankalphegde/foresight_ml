"""
Production SEC XBRL ingestion.
Handles:
- Initial full load
- Incremental updates
- Amendment-safe refresh (last N quarters)
"""

import os
import pandas as pd
from google.cloud import storage
from datetime import datetime

from src.data.clients.sec_client import SECClient
from src.data.clients.sec_xbrl_client import SECXBRLClient

BUCKET = os.environ["GCS_BUCKET"]
USER_AGENT = os.environ["SEC_USER_AGENT"]
REFRESH_LAST_N_QUARTERS = 8
RAW_PREFIX = "raw/sec_xbrl"


def quarter_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["fiscal_year"].astype(int).astype(str)
        + "_"
        + df["fiscal_period"]
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    df["fiscal_period"] = df["fiscal_period"].astype(str)
    df = df[df["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])]
    df = df.dropna(subset=["fiscal_year"])
    df["quarter_key"] = quarter_key(df)
    return df


def load_existing(storage_client, cik):
    blob_path = f"{RAW_PREFIX}/cik={cik}/data.parquet"
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    with blob.open("rb") as f:
        return pd.read_parquet(f)


def save(storage_client, cik, df):
    blob_path = f"{RAW_PREFIX}/cik={cik}/data.parquet"
    bucket = storage_client.bucket(BUCKET)
    with bucket.blob(blob_path).open("wb") as f:
        df.to_parquet(f, index=False)
    print(f"Saved -> {blob_path}")


def main():
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(BUCKET)

    companies_df = pd.read_csv(
        bucket.blob("reference/companies.csv").open("r")
    )

    sec = SECClient(user_agent=USER_AGENT)
    xbrl = SECXBRLClient(sec)

    total_companies = len(companies_df)
    print(f"Processing {total_companies} companies...")

    for idx, row in companies_df.iterrows():
        cik = str(row["cik"]).zfill(10)
        print(f"[{idx+1}/{total_companies}] Processing {cik}")

        try:
            new_df = xbrl.extract_long_format(cik)
        except Exception as e:
            print(f"Skipping {cik} due to error: {e}")
            continue
        if new_df.empty:
            continue

        new_df = clean(new_df)

        existing_df = load_existing(storage_client, cik)

        if existing_df is None:
            print("Full historical load.")
            save(storage_client, cik, new_df)
            continue

        existing_df = clean(existing_df)

        # If old data does not have quarter_end_date, derive it
        if "quarter_end_date" not in existing_df.columns:
            quarter_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
            existing_df["quarter_end_date"] = pd.to_datetime(
                existing_df["fiscal_year"].astype(int).astype(str)
                + "-"
                + existing_df["fiscal_period"].map(quarter_map)
            )
        else:
            existing_df["quarter_end_date"] = pd.to_datetime(
                existing_df["quarter_end_date"]
            )

        # Identify most recent N quarters
        latest_quarters = (
            existing_df[["quarter_key", "quarter_end_date"]]
            .drop_duplicates()
            .sort_values("quarter_end_date")
            .tail(REFRESH_LAST_N_QUARTERS)["quarter_key"]
        )

        # REMOVE those recent quarters from existing
        existing_df = existing_df[
            ~existing_df["quarter_key"].isin(latest_quarters)
        ]

        # Identify quarters already safely stored
        old_quarters = set(existing_df["quarter_key"].unique())

        # Keep:
        # - new quarters
        # - refreshed quarters
        incremental_df = new_df[
            ~new_df["quarter_key"].isin(old_quarters)
        ]

        updated = pd.concat([existing_df, incremental_df], ignore_index=True)

        # Optional dedup safety
        updated = updated.drop_duplicates(
            subset=["cik", "fiscal_year", "fiscal_period", "tag"],
            keep="last"
        )

        save(storage_client, cik, updated)

if __name__ == "__main__":
    main()