"""Main FastAPI application setup."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import gcsfs
import joblib
import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import alerts, company, drift, health, predict

logger = logging.getLogger(__name__)

# This dictionary will hold our loaded model and scaler in memory
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the ML model and scaler on startup."""
    logger.info("Starting up API and loading models...")
    try:
        # 1. Load the production model from MLflow
        model_name = "foresight_xgboost"
        model_uri = f"models:/{model_name}/Production"
        ml_models["model"] = mlflow.pyfunc.load_model(model_uri)

        # 2. Load the scaler directly from GCS
        fs = gcsfs.GCSFileSystem()
        scaler_path = "financial-distress-data/models/v1.0/scaler_pipeline.pkl"
        with fs.open(scaler_path, "rb") as f:
            ml_models["scaler"] = joblib.load(f)

        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")

    yield
    # Clean up when the server shuts down
    ml_models.clear()


app = FastAPI(
    title="Foresight-ML API",
    description="Inference and monitoring API for the Financial Distress model.",
    lifespan=lifespan,
)

# Allow the Dashboard to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect all the router files
app.include_router(predict.router)
app.include_router(company.router)
app.include_router(alerts.router)
app.include_router(health.router)
app.include_router(drift.router)
