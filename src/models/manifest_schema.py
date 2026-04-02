"""Pydantic schema for manifest.json — the inference provenance certificate.

Every batch-inference run writes a ``manifest.json`` alongside its
``scores.parquet``.  This module defines the exact shape and validation
rules for that JSON file.

Why Pydantic instead of raw ``json.dumps(dict)``?
- A typo in a field name or a missing field silently produces invalid JSON.
- Pydantic validates at WRITE time: if you forget ``scored_at`` or pass
  ``roc_auc=1.5``, you get a ``ValidationError`` immediately.
- Pydantic is already a project dependency (``pyproject.toml``).
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


class ManifestSchema(BaseModel):
    """Schema for manifest.json that accompanies every scores.parquet batch.

    Organized into four logical sections:
        1. Identity  — what model is this?
        2. Lineage   — where did it come from?
        3. Quality   — how good is it and what settings were used?
        4. Output    — what did it produce?
    """

    # --- Section 1: Identity ---
    schema_version: str = Field(
        default="1.0",
        description="Schema format version for forward-compatible evolution.",
        pattern=r"^\d+\.\d+$",
    )
    model_name: str = Field(
        ...,
        description="MLflow registered model name.",
        min_length=1,
        examples=["foresight_xgboost"],
    )
    model_version: str = Field(
        ...,
        description="Semantic version tag from MLflow registry.",
        pattern=r"^v\d+",
        examples=["v1", "v2"],
    )

    # --- Section 2: Lineage ---
    mlflow_run_id: str = Field(
        ...,
        description="MLflow run ID that trained this model.",
        min_length=1,
    )
    trained_at: datetime = Field(
        ...,
        description="UTC timestamp when the model was trained.",
    )
    scored_at: datetime = Field(
        ...,
        description="UTC timestamp when this batch inference ran.",
    )

    # --- Section 3: Quality & Config ---
    roc_auc: float = Field(
        ...,
        description="Test ROC-AUC metric from the training run.",
        ge=0.0,
        le=1.0,
    )
    prediction_horizon: int = Field(
        ...,
        description="Number of quarters ahead the label predicts.",
        ge=1,
    )
    features_used: List[str] = Field(
        ...,
        description="Pre-dummy raw feature columns the model was trained on.",
        min_length=1,
    )

    # --- Section 4: Output Metadata ---
    row_count: int = Field(
        ...,
        description="Number of rows in the output scores.parquet.",
        gt=0,
    )
    gcs_scores_path: str = Field(
        ...,
        description="Full GCS URI to the scores.parquet file.",
    )
    inference_duration_seconds: float = Field(
        ...,
        description="Wall-clock time for the inference batch in seconds.",
        ge=0.0,
    )

    @field_validator("gcs_scores_path")
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Enforce GCS URI format."""
        if not v.startswith("gs://"):
            raise ValueError("gcs_scores_path must start with gs://")
        if not v.endswith(".parquet"):
            raise ValueError("gcs_scores_path must end with .parquet")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_version": "1.0",
                    "model_name": "foresight_xgboost",
                    "model_version": "v1",
                    "mlflow_run_id": "abc123def456",
                    "trained_at": "2026-03-15T10:00:00+00:00",
                    "scored_at": "2026-03-30T06:00:00+00:00",
                    "roc_auc": 0.9769,
                    "prediction_horizon": 1,
                    "features_used": ["total_assets", "net_income"],
                    "row_count": 48291,
                    "gcs_scores_path": "gs://bucket/inference/v1/scores.parquet",
                    "inference_duration_seconds": 42.7,
                }
            ]
        }
    }
