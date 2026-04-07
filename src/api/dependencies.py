"""API dependencies and rate limiting."""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Sets up the rate limiter (100 requests per minute) for later
limiter = Limiter(key_func=get_remote_address)

async def verify_api_key(request: Request):
    """Mock authentication for Day 1.

    We will implement the real GCP Secret Manager logic here later.
    """
    pass