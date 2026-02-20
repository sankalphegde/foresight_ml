"""SEC EDGAR API client with caching and Pydantic validation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, field_validator


class Filing(BaseModel):
    """SEC filing model."""

    cik: str
    ticker: str | None = None
    form: str
    filing_date: str
    accession_number: str = Field(..., alias="accessionNumber")

    @field_validator("filing_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Ensure date is in YYYY-MM-DD format."""
        if len(v) != 10 or v[4] != "-" or v[7] != "-":
            raise ValueError(f"Invalid date format: {v}")
        return v


class CompanyFilings(BaseModel):
    """Company filings response model."""

    cik: str
    name: str = Field(..., alias="name")
    filings: dict


class SECClient:
    """Minimal SEC EDGAR client with rate limiting and caching."""

    BASE_URL = "https://data.sec.gov"

    def __init__(self, user_agent: str, cache_dir: str = "cache/sec") -> None:
        """Initialize SEC client.

        Args:
            user_agent: User agent with email (required by SEC)
            cache_dir: Directory for caching responses

        Raises:
            ValueError: If user agent is invalid
        """
        if not user_agent or "@" not in user_agent:
            raise ValueError("User agent must include email: 'Name email@example.com'")

        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request = 0.0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _rate_limit(self) -> None:
        """Enforce 10 req/s SEC limit."""
        elapsed = time.time() - self.last_request
        if elapsed < 0.1:  # 100ms = 10 req/s
            time.sleep(0.1 - elapsed)
        self.last_request = time.time()

    def _cache_key(self, url: str) -> str:
        """Generate cache filename from URL."""
        return hashlib.md5(url.encode()).hexdigest() + ".json"

    def _get_cached(self, url: str) -> dict | None:
        """Get cached response if exists."""
        try:
            cache_file = self.cache_dir / self._cache_key(url)
            if cache_file.exists():
                with open(cache_file) as f:
                    data: dict[Any, Any] = json.load(f)
                    return data
        except (OSError, IOError) as e:
            # Silently ignore cache read errors (e.g., I/O errors on Windows Docker mounts)
            pass
        return None

    def _save_cache(self, url: str, data: dict) -> None:
        """Save response to cache."""
        try:
            cache_file = self.cache_dir / self._cache_key(url)
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except (OSError, IOError) as e:
            # Silently ignore cache write errors (e.g., I/O errors on Windows Docker mounts)
            pass

    def get(self, endpoint: str, use_cache: bool = True) -> dict:
        """GET request with caching and rate limiting.

        Args:
            endpoint: API endpoint path
            use_cache: Whether to use cached responses

        Returns:
            Response JSON data

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.BASE_URL}{endpoint}"

        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                return cached

        # Rate limit and fetch
        self._rate_limit()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        data: dict[Any, Any] = response.json()

        # Cache result
        if use_cache:
            self._save_cache(url, data)

        return data

    def get_company_filings(self, cik: str) -> CompanyFilings:
        """Get company filings metadata.

        Args:
            cik: Company CIK number

        Returns:
            Validated company filings data
        """
        cik_padded = str(cik).zfill(10)
        data = self.get(f"/submissions/CIK{cik_padded}.json")
        return CompanyFilings(**data)

    def filter_filings(
        self,
        filings_data: CompanyFilings,
        form_types: list[str],
        start_date: str | None = None,
    ) -> list[Filing]:
        """Extract specific filing types from filings data.

        Args:
            filings_data: Company filings data
            form_types: List of form types to filter (e.g., ["10-K", "10-Q"])
            start_date: Optional start date filter (YYYY-MM-DD)

        Returns:
            List of validated filings
        """
        recent = filings_data.filings.get("recent", {})
        results = []

        for i in range(len(recent.get("accessionNumber", []))):
            if recent["form"][i] in form_types:
                if start_date and recent["filingDate"][i] < start_date:
                    continue
                filing = Filing(
                    cik=filings_data.cik,
                    form=recent["form"][i],
                    filing_date=recent["filingDate"][i],
                    accessionNumber=recent["accessionNumber"][i],
                )
                results.append(filing)

        return results
