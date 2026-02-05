import os
import json
from datetime import datetime

import pandas as pd
from google.cloud import storage

from src.data.clients.sec_client import SECClient


def get_year_quarter(execution_date: str) -> tuple[int, str]:
    dt = datetime.fromisoformat(execution_date)
    quarter = (dt.month - 1) // 3 + 1
    return dt.year, f"Q{quarter}"


def main() -> None:
    # ---- REQUIRED ENV VARS ----
    execution_date = os.environ["EXECUTION_DATE"]
    bucket_name = os.environ["GCS_BUCKET"]
    user_agent = os.environ["SEC_USER_AGENT"]

    year, quarter = get_year_quarter(execution_date)

    print("Starting SEC ingestion:", year, quarter)

    # ---- GCS CLIENT ----
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # ---- INPUT: COMPANY LIST ----
    companies_blob = bucket.blob("reference/companies.csv")
    companies_df = pd.read_csv(companies_blob.open("r"))

    # 🚧 SAFETY GUARD (REMOVE LATER)
    companies_df = companies_df.head(100)

    # ---- SEC CLIENT ----
    sec_client = SECClient(user_agent=user_agent)

    # ---- OUTPUT (JSONL) ----
    output_path = f"raw/sec/year={year}/quarter={quarter}/filings.jsonl"
    blob = bucket.blob(output_path)

    total_written = 0

    # Collect all records first, then write once
    records = []
    for _, row in companies_df.iterrows():
        cik = str(row["cik"]).zfill(10)
        ticker = row["ticker"]

        try:
            filings_data = sec_client.get_company_filings(cik)

            filings = sec_client.filter_filings(
                filings_data,
                form_types=["10-K", "10-Q"],
                start_date="2020-01-01",
            )

            for filing in filings:
                record = filing.model_dump()
                record["ticker"] = ticker
                record["cik"] = cik

                records.append(json.dumps(record) + "\n")
                total_written += 1

        except Exception as e:
            # Never kill the whole job for one bad company
            print(f"Failed for {ticker} ({cik}): {e}")

    # Write all records at once
    if records:
        with blob.open("w") as f:
            f.write("".join(records))

    print(
        f"Wrote {total_written} SEC filings "
        f"to gs://{bucket_name}/{output_path}"
    )


if __name__ == "__main__":
    main()
