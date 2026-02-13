"""Core data pipeline functions for fetching and merging SEC and FRED data."""

import json
import os
from pathlib import Path

import pandas as pd

from src.data.clients.fred_client import FREDClient
from src.data.clients.sec_client import SECClient

# Environment config
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
FRED_API_KEY = os.getenv("FRED_API_KEY")
COMPANIES_CSV = os.getenv("COMPANIES_CSV", "data/companies.csv")
MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "50"))


def load_companies() -> list[dict]:
    """Load companies from CSV file."""
    df = pd.read_csv(COMPANIES_CSV).head(MAX_COMPANIES)

    return [
        {
            "ticker": row["ticker"],
            "cik": str(row["cik"]).zfill(10),
        }
        for _, row in df.iterrows()
    ]


def fetch_sec_data(output_dir: str = "/tmp") -> str:
    """Fetch SEC filings and write them to disk."""
    if not SEC_USER_AGENT:
        raise ValueError("SEC_USER_AGENT environment variable not set")

    client = SECClient(
        user_agent=SEC_USER_AGENT,
        cache_dir=f"{output_dir}/cache/sec",
    )

    companies = load_companies()
    all_filings: list[dict] = []

    for company in companies:
        filings_data = client.get_company_filings(company["cik"])
        filings = client.filter_filings(
            filings_data,
            form_types=["10-K", "10-Q"],
            start_date="2020-01-01",
        )

        for filing in filings:
            data = filing.model_dump()
            data["ticker"] = company["ticker"]
            all_filings.append(data)

    output_path = Path(output_dir) / "sec_filings.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_filings, f)

    return str(output_path)


def fetch_fred_data(output_dir: str = "/tmp") -> str:
    """Fetch FRED indicators and write them to disk."""
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable not set")

    client = FREDClient(
        api_key=FRED_API_KEY,
        cache_dir=f"{output_dir}/cache/fred",
    )

    df = client.get_common_indicators(
        start_date="2020-01-01",
        frequency="q",
    )

    output_path = Path(output_dir) / "fred_indicators.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    return str(output_path)


def merge_data(
    sec_path: str,
    fred_path: str,
    output_dir: str = "/tmp",
) -> str:
    """Merge SEC and FRED datasets."""
    with open(sec_path) as f:
        sec_data = json.load(f)

    fred_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)

    sec_df = pd.DataFrame(sec_data)
    sec_df["filing_date"] = pd.to_datetime(sec_df["filing_date"])
    sec_df["quarter"] = sec_df["filing_date"].dt.to_period("Q")

    fred_df["quarter"] = fred_df.index.to_period("Q")

    merged = sec_df.merge(
        fred_df.reset_index(),
        on="quarter",
        how="left",
    )

    output_path = Path(output_dir) / "merged_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(output_path)
    return str(output_path)
