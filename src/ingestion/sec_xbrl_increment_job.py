"""Production SEC XBRL ingestion.

Handles:
- Initial full load
- Incremental updates
- Amendment-safe refresh (last N quarters)
"""

import os

import pandas as pd
from google.cloud import storage

from src.data.clients.sec_client import SECClient
from src.data.clients.sec_xbrl_client import SECXBRLClient

REFRESH_LAST_N_QUARTERS = 8
RAW_PREFIX = "raw/sec_xbrl"


def quarter_key(df: pd.DataFrame) -> pd.Series:
    """Generate a combined string key for the fiscal quarter."""
    return df["fiscal_year"].astype(int).astype(str) + "_" + df["fiscal_period"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format SEC XBRL data columns."""
    df = df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    df["fiscal_period"] = df["fiscal_period"].astype(str)
    df = df[df["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])]
    df = df.dropna(subset=["fiscal_year"])
    df["quarter_key"] = quarter_key(df)
    return df


def load_existing(
    storage_client: "storage.Client", bucket_name: str, cik: str
) -> "pd.DataFrame | None":
    """Load existing SEC data for a given CIK from cloud storage."""
    blob_path = f"{RAW_PREFIX}/cik={cik}/data.parquet"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    with blob.open("rb") as f:
        return pd.read_parquet(f)


def save(storage_client: "storage.Client", bucket_name: str, cik: str, df: "pd.DataFrame") -> None:
    """Save updated SEC data for a given CIK to cloud storage."""
    blob_path = f"{RAW_PREFIX}/cik={cik}/data.parquet"
    bucket = storage_client.bucket(bucket_name)
    with bucket.blob(blob_path).open("wb") as f:
        df.to_parquet(f, index=False)
    print(f"Saved -> {blob_path}")


def main() -> None:
    """Execute the main ingestion pipeline for SEC XBRL data."""
    bucket_name = os.environ.get("GCS_BUCKET")
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not bucket_name:
        raise RuntimeError("Missing required environment variable: GCS_BUCKET")
    if not user_agent:
        raise RuntimeError("Missing required environment variable: SEC_USER_AGENT")

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    companies_df = pd.read_csv(bucket.blob("reference/companies.csv").open("r"))
    companies_df = companies_df.head(5)

    sec = SECClient(user_agent=user_agent)
    xbrl = SECXBRLClient(sec)

    total_companies = len(companies_df)
    print(f"Processing {total_companies} companies...")

    for idx, row in companies_df.iterrows():
        cik = str(row["cik"]).zfill(10)
        print(f"[{idx + 1}/{total_companies}] Processing {cik}")
        try:
            new_df = xbrl.extract_long_format(cik)
        except Exception as e:
            print(f"Skipping {cik} due to error: {e}")
            continue
        if new_df.empty:
            continue

        new_df = clean(new_df)

        existing_df = load_existing(storage_client, bucket_name, cik)

        if existing_df is None:
            print("Full historical load.")
            save(storage_client, bucket_name, cik, new_df)
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
            existing_df["quarter_end_date"] = pd.to_datetime(existing_df["quarter_end_date"])

        # Identify most recent N quarters
        latest_quarters = (
            existing_df[["quarter_key", "quarter_end_date"]]
            .drop_duplicates()
            .sort_values("quarter_end_date")
            .tail(REFRESH_LAST_N_QUARTERS)["quarter_key"]
        )

        # REMOVE those recent quarters from existing
        existing_df = existing_df[~existing_df["quarter_key"].isin(latest_quarters)]

        # Identify quarters already safely stored
        old_quarters = set(existing_df["quarter_key"].unique())

        # Keep:
        # - new quarters
        # - refreshed quarters
        incremental_df = new_df[~new_df["quarter_key"].isin(old_quarters)]

        updated = pd.concat([existing_df, incremental_df], ignore_index=True)

        # Optional dedup safety
        updated = updated.drop_duplicates(
            subset=["cik", "fiscal_year", "fiscal_period", "tag"], keep="last"
        )

        save(storage_client, bucket_name, cik, updated)


if __name__ == "__main__":
    main()
