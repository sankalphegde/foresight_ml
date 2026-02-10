"""
Basic tests for API clients.
Run: pytest test_data_ingestion.py -v
"""
import pytest

from src.data.clients.fred_client import FREDClient
from src.data.clients.sec_client import SECClient


def test_sec_client_init() -> None:
    """Test SEC client initialization."""
    client = SECClient(user_agent="Test test@example.com")
    assert client.session.headers['User-Agent'] == "Test test@example.com"
    assert client.cache_dir.exists()


def test_sec_client_invalid_user_agent() -> None:
    """Test SEC client rejects invalid user agent."""
    with pytest.raises(ValueError, match="User agent must include email"):
        SECClient(user_agent="invalid")


def test_fred_client_init() -> None:
    """Test FRED client initialization."""
    client = FREDClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.cache_dir.exists()


def test_cache_key_generation() -> None:
    """Test cache file path generation."""
    client = SECClient(user_agent="Test test@example.com")
    path1 = client._cache_key("test")
    path2 = client._cache_key("test")

    # Same input should generate same path
    assert path1 == path2

    # Different input should generate different path
    path3 = client._cache_key("different")
    assert path1 != path3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
