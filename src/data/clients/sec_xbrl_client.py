"""SEC XBRL Company Facts client (fully extracted long format)."""

from typing import Any

import pandas as pd

from src.data.clients.sec_client import SECClient


class SECXBRLClient:
    """Client for SEC XBRL company facts endpoint."""

    def __init__(self, sec_client: SECClient) -> None:
        """Initialize SECXBRLClient.

        Args:
            sec_client: Base SEC client used to perform HTTP requests.
        """
        self.sec_client = sec_client

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch full company facts JSON."""
        cik_padded = str(cik).zfill(10)
        return self.sec_client.get(f"/api/xbrl/companyfacts/CIK{cik_padded}.json")

    def extract_long_format(
        self,
        cik: str,
        forms: list[str] | None = None,
        only_quarters: bool = True,
    ) -> pd.DataFrame:
        """Extract ALL us-gaap facts in long format.

        Output schema:

        | cik | tag | unit | value | start_date | end_date |
        | fiscal_year | fiscal_period | form | filed_date | frame |
        """
        forms = forms or ["10-Q", "10-K"]

        data = self.get_company_facts(cik)
        facts = data.get("facts", {}).get("us-gaap", {})

        records: list[dict[str, Any]] = []

        for tag_name, tag_data in facts.items():
            units = tag_data.get("units", {})

            for unit_name, unit_entries in units.items():
                for entry in unit_entries:
                    form = entry.get("form")
                    if form not in forms:
                        continue

                    if only_quarters:
                        if entry.get("fp") not in ["Q1", "Q2", "Q3", "Q4"]:
                            continue

                    records.append(
                        {
                            "cik": str(cik).zfill(10),
                            "tag": tag_name,
                            "unit": unit_name,
                            "value": entry.get("val"),
                            "start_date": entry.get("start"),
                            "end_date": entry.get("end"),
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "form": form,
                            "filed_date": entry.get("filed"),
                            "frame": entry.get("frame"),
                        }
                    )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # ---- Type enforcement (important for parquet & BigQuery) ----
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

        # Optional: enforce datetime
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")

        return df
