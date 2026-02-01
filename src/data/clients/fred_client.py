"""FRED API client with caching and Pydantic validation."""
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel, Field


class SeriesObservation(BaseModel):
    """FRED series observation model."""

    date: str
    value: str


class SeriesResponse(BaseModel):
    """FRED API response model."""

    observations: list[SeriesObservation]


class FREDClient:
    """Minimal FRED client with caching."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Common economic indicators
    INDICATORS = {
        "fed_funds": "DFF",
        "inflation": "CPIAUCSL",
        "unemployment": "UNRATE",
        "gdp": "GDP",
        "credit_spread": "BAMLC0A4CBBB",
        "vix": "VIXCLS",
    }

    def __init__(self, api_key: str, cache_dir: str = "cache/fred") -> None:
        """Initialize FRED client.

        Args:
            api_key: FRED API key
            cache_dir: Directory for caching responses
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _cache_key(self, params: dict) -> str:
        """Generate cache filename from params."""
        key_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest() + ".json"

    def _get_cached(self, params: dict) -> Optional[dict]:
        """Get cached response if exists."""
        cache_file = self.cache_dir / self._cache_key(params)
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_cache(self, params: dict, data: dict) -> None:
        """Save response to cache."""
        cache_file = self.cache_dir / self._cache_key(params)
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        frequency: Optional[str] = None,
        use_cache: bool = True,
    ) -> SeriesResponse:
        """Get time series data.

        Args:
            series_id: FRED series ID
            start_date: Optional start date (YYYY-MM-DD)
            frequency: Optional frequency (d, w, m, q, a)
            use_cache: Whether to use cached responses

        Returns:
            Validated series response

        Raises:
            requests.HTTPError: If request fails
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if frequency:
            params["frequency"] = frequency

        # Check cache
        if use_cache:
            cached = self._get_cached(params)
            if cached:
                return SeriesResponse(**cached)

        # Fetch from API
        response = self.session.get(
            f"{self.BASE_URL}/series/observations", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Cache result
        if use_cache:
            self._save_cache(params, data)

        return SeriesResponse(**data)

    def get_series_dataframe(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get series as pandas DataFrame.

        Args:
            series_id: FRED series ID
            start_date: Optional start date
            frequency: Optional frequency

        Returns:
            DataFrame with date index and value column
        """
        response = self.get_series(series_id, start_date, frequency)

        df = pd.DataFrame([obs.model_dump() for obs in response.observations])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df.set_index("date")

    def get_common_indicators(
        self, start_date: Optional[str] = None, frequency: str = "q"
    ) -> pd.DataFrame:
        """Get all common indicators as single DataFrame.

        Args:
            start_date: Optional start date
            frequency: Data frequency (default: quarterly)

        Returns:
            DataFrame with all common indicators
        """
        dfs = {}

        for name, series_id in self.INDICATORS.items():
            try:
                df = self.get_series_dataframe(series_id, start_date, frequency)
                dfs[name] = df["value"]
            except Exception as e:
                print(f"Failed to fetch {name}: {e}")

        return pd.DataFrame(dfs)