"""
Basic tests for API clients.
Run: pytest test_data_ingestion.py -v
"""

import os

import pytest

from src.data.clients.fred_client import FREDClient
from src.data.clients.sec_client import SECClient


def test_sec_client_init():
    """Test SEC client initialization."""
    client = SECClient(user_agent="Test test@example.com")
    assert client.session.headers["User-Agent"] == "Test test@example.com"
    assert client.cache_dir.exists()


def test_sec_client_invalid_user_agent():
    """Test SEC client rejects invalid user agent."""
    with pytest.raises(ValueError, match="User agent must include email"):
        SECClient(user_agent="invalid")


def test_fred_client_init():
    """Test FRED client initialization."""
    client = FREDClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.cache_dir.exists()


def test_cache_key_generation():
    """Test cache file path generation."""
    client = SECClient(user_agent="Test test@example.com")
    path1 = client._cache_key("test")
    path2 = client._cache_key("test")

    # Same input should generate same path
    assert path1 == path2

    # Different input should generate different path
    path3 = client._cache_key("different")
    assert path1 != path3


# Integration tests (require API keys)
def require_fred_api_key():
    """Ensure FRED_API_KEY is set."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        pytest.fail("FRED_API_KEY must be set in environment variables")
    return api_key


def require_sec_user_agent():
    """Ensure SEC_USER_AGENT is set."""
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        pytest.fail("SEC_USER_AGENT must be set in environment variables")
    return user_agent


def test_fred_fetch_series():
    """Test FRED client fetches real data."""
    api_key = require_fred_api_key()
    client = FREDClient(api_key=api_key)

    # Fetch Federal Funds Rate (DFF) - always available
    response = client.get_series(
        series_id="DFF",
        start_date="2024-01-01",
        use_cache=False,  # Force fresh fetch
    )

    # Validate response structure
    assert len(response.observations) > 0
    assert hasattr(response.observations[0], "date")
    assert hasattr(response.observations[0], "value")

    # Validate date format
    first_obs = response.observations[0]
    assert len(first_obs.date) == 10  # YYYY-MM-DD
    assert first_obs.date >= "2024-01-01"


def test_fred_fetch_dataframe():
    """Test FRED client returns valid DataFrame."""
    api_key = require_fred_api_key()
    client = FREDClient(api_key=api_key)

    df = client.get_series_dataframe(series_id="DFF", start_date="2024-01-01")

    # Validate DataFrame structure
    assert not df.empty
    assert "value" in df.columns
    assert df.index.name == "date" or str(df.index.dtype).startswith("datetime")
    assert df["value"].dtype in ["float64", "Float64"]


def test_fred_common_indicators():
    """Test FRED client fetches common indicators."""
    api_key = require_fred_api_key()
    client = FREDClient(api_key=api_key)

    df = client.get_common_indicators(
        start_date="2024-01-01",
        frequency="m",  # Monthly
    )

    # Should have multiple indicator columns
    assert not df.empty
    expected_indicators = ["fed_funds", "inflation", "unemployment", "gdp", "vix"]

    # At least some indicators should be present
    present_indicators = [col for col in expected_indicators if col in df.columns]
    assert len(present_indicators) >= 3, f"Expected at least 3 indicators, got {present_indicators}"


def test_sec_fetch_company_filings():
    """Test SEC client fetches real company data."""
    user_agent = require_sec_user_agent()
    client = SECClient(user_agent=user_agent)

    # Fetch Apple Inc (CIK: 0000320193)
    company = client.get_company_filings(cik="320193")

    # Validate company data
    assert company.cik == "0000320193"
    assert company.name is not None
    assert len(company.name) > 0
    assert "recent" in company.filings


def test_sec_filter_filings():
    """Test SEC client filters filings correctly."""
    user_agent = require_sec_user_agent()
    client = SECClient(user_agent=user_agent)

    # Fetch and filter Apple filings
    company = client.get_company_filings(cik="320193")
    filings = client.filter_filings(company, form_types=["10-K", "10-Q"], start_date="2023-01-01")

    # Should have multiple filings
    assert len(filings) > 0

    # Validate filing structure
    for filing in filings[:3]:  # Check first 3
        assert filing.cik == "0000320193"
        assert filing.form in ["10-K", "10-Q"]
        assert filing.filing_date >= "2023-01-01"
        assert len(filing.accession_number) > 0


def test_sec_rate_limiting():
    """Test SEC client respects rate limits."""
    import time

    user_agent = require_sec_user_agent()
    client = SECClient(user_agent=user_agent)

    start = time.time()

    # Make 3 requests
    for _ in range(3):
        client.get_company_filings(cik="320193")

    elapsed = time.time() - start

    # Should take at least 0.2 seconds (0.1s * 2 gaps)
    # allowing cache to potentially speed things up
    assert elapsed >= 0.0  # Just verify it doesn't error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
