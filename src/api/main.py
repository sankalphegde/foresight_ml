"""Main FastAPI application setup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Import our dependencies and routers
from src.api.dependencies import limiter
from src.api.routers import alerts, company, drift, health, predict

# Initialize FastAPI App
app = FastAPI(
    title="Foresight-ML API",
    description="Serving API for the XGBoost Financial Distress Model",
    version="1.0.0",
)

# Setup SlowAPI Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

# Setup CORS so the Streamlit dashboard can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all Routers
app.include_router(predict.router, tags=["Prediction"])
app.include_router(company.router, tags=["Company Data"])
app.include_router(alerts.router, tags=["Watchlist Alerts"])
app.include_router(health.router, tags=["System Health"])
app.include_router(drift.router, tags=["Model Monitoring"])
