"""SEC XBRL ingestion job.

Fetches SEC XBRL company facts data and uploads processed output to GCS.
"""

import json
import os
from datetime import datetime

import pandas as pd
from google.cloud import storage

from src.data.clients.sec_client import SECClient
from src.data.clients.sec_xbrl_client import SECXBRLClient


def get_year_quarter(execution_date: str) -> tuple[int, str]:
    """Extract year and quarter string from an ISO execution date.

    Args:
        execution_date: Execution date in ISO format (YYYY-MM-DD).

    Returns:
        A tuple of (year, quarter_string) where quarter_string is like "Q1".
    """
    dt = datetime.fromisoformat(execution_date)
    quarter = (dt.month - 1) // 3 + 1
    return dt.year, f"Q{quarter}"


def main() -> None:
    """Entry point for SEC XBRL ingestion job.

    Reads execution parameters from environment variables,
    fetches SEC data, and writes output to the configured GCS bucket.
    """
    execution_date = os.environ["EXECUTION_DATE"]
    bucket_name = os.environ["GCS_BUCKET"]
    user_agent = os.environ["SEC_USER_AGENT"]

    year, quarter = get_year_quarter(execution_date)

    print("Starting SEC XBRL ingestion:", year, quarter)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    companies_blob = bucket.blob("reference/companies.csv")
    companies_df = pd.read_csv(companies_blob.open("r"))

    # 🚧 limit for testing
    companies_df = companies_df.head(100)

    sec_client = SECClient(user_agent=user_agent)
    xbrl_client = SECXBRLClient(sec_client)

    total_rows = 0

    for _, row in companies_df.iterrows():
        cik = str(row["cik"]).zfill(10)
        ticker = row["ticker"]

        try:
            print(f"Processing {ticker} ({cik})")

            # ---------- 1. STORE RAW JSON ----------
            raw_json = xbrl_client.get_company_facts(cik)

            raw_path = f"raw/sec_xbrl_json/year={year}/quarter={quarter}/cik={cik}.json"

            bucket.blob(raw_path).upload_from_string(
                json.dumps(raw_json),
                content_type="application/json",
            )

            # ---------- 2. EXTRACT LONG FORMAT ----------
            df = xbrl_client.extract_long_format(cik)

            if df.empty:
                continue

            df["ticker"] = ticker

            # ---------- 3. WRITE PARQUET PER COMPANY ----------
            parquet_path = f"raw/sec_xbrl_long/year={year}/quarter={quarter}/cik={cik}.parquet"

            with bucket.blob(parquet_path).open("wb") as f:
                df.to_parquet(f, index=False)

            total_rows += len(df)

        except Exception as e:
            print(f"Failed XBRL for {ticker} ({cik}): {e}")

    print(
        f"Finished ingestion. Wrote {total_rows} total long-format rows "
        f"to gs://{bucket_name}/raw/sec_xbrl_long/year={year}/quarter={quarter}/"
    )


if __name__ == "__main__":
    main()
