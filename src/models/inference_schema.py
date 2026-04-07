"""Pydantic schemas and validators for the inference input/output contract.

This module defines the formal interface between the data pipeline and the
model.  It enforces that:

1. Input data has the required identity columns and no label leakage.
2. All feature columns are numeric (pre-get_dummies validation).
3. Output scores contain versioning metadata for auditability.

The validation approach is *structural* rather than a hardcoded feature
list because ``train.py`` uses ``pd.get_dummies(dummy_na=True)`` which
creates a dynamic column set depending on the data.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

# Identity columns — required for downstream joins, NOT features.
IDENTITY_COLUMNS: list[str] = ["firm_id", "fiscal_year", "fiscal_period"]

# The label column — must NOT be present in inference input.
LABEL_COLUMN: str = "distress_label"


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------


class InferenceOutputRow(BaseModel):
    """Schema for a single row of ``scores.parquet`` output.

    Includes the prediction, confidence intervals (optional), and
    versioning metadata columns injected by ``predict.py``.
    """

    # --- Identity ---
    firm_id: str = Field(...)
    fiscal_year: int = Field(...)
    fiscal_period: str = Field(...)

    # --- Prediction ---
    distress_probability: float = Field(
        ...,
        description="Model's predicted probability of financial distress.",
        ge=0.0,
        le=1.0,
    )

    # --- Confidence Intervals (optional — simple ±5% margin) ---
    confidence_interval_lower: float | None = Field(
        None,
        description="Lower bound of confidence interval.",
        ge=0.0,
        le=1.0,
    )
    confidence_interval_upper: float | None = Field(
        None,
        description="Upper bound of confidence interval.",
        ge=0.0,
        le=1.0,
    )

    # --- Versioning Metadata ---
    model_version: str = Field(
        ...,
        description="Version tag from MLflow registry (e.g. 'v1').",
        pattern=r"^v\d+",
    )
    mlflow_run_id: str = Field(
        ...,
        description="MLflow training run ID.",
        min_length=1,
    )
    trained_at: str = Field(
        ...,
        description="ISO 8601 timestamp of model training.",
    )
    scored_at: str = Field(
        ...,
        description="ISO 8601 timestamp of this inference batch.",
    )
    model_roc_auc: float = Field(
        ...,
        description="Test ROC-AUC of the model at training time.",
        ge=0.0,
        le=1.0,
    )


# ---------------------------------------------------------------------------
# DataFrame-level Validation Helpers
# ---------------------------------------------------------------------------


def validate_inference_input(df: pd.DataFrame) -> list[str]:  # noqa: F821
    """Validate input DataFrame before prediction.

    Checks structural rules rather than a hardcoded feature list,
    because ``train.py`` uses ``pd.get_dummies()`` which creates
    dynamic columns.

    Returns a list of error messages.  Empty list means valid.

    Args:
        df: Input DataFrame to validate.

    Returns:
        List of human-readable error strings.
    """
    errors: list[str] = []

    # Identity columns must exist
    for col in IDENTITY_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing identity column: {col}")

    # Label should NOT be present in inference input
    if LABEL_COLUMN in df.columns:
        errors.append(
            f"Label column '{LABEL_COLUMN}' found in inference input — " "remove it before scoring"
        )

    # Feature columns (everything except identity) must be numeric
    feature_cols = [c for c in df.columns if c not in IDENTITY_COLUMNS]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Non-numeric feature column: '{col}' ({df[col].dtype})")

    # All-null columns indicate upstream pipeline bugs
    for col in feature_cols:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' is entirely null — likely a pipeline issue")

    return errors


def validate_inference_output(df: pd.DataFrame) -> list[str]:  # noqa: F821
    """Validate output scores DataFrame.

    Args:
        df: Output scores DataFrame to validate.

    Returns:
        List of human-readable error strings.
    """
    errors: list[str] = []

    required = [
        "firm_id",
        "distress_probability",
        "model_version",
        "mlflow_run_id",
        "scored_at",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing output columns: {missing}")

    # Prediction range check
    if "distress_probability" in df.columns:
        oob = df[(df["distress_probability"] < 0) | (df["distress_probability"] > 1)]
        if len(oob) > 0:
            errors.append(f"{len(oob)} rows with distress_probability outside [0, 1]")

    # Versioning columns should never be null
    for col in ["model_version", "mlflow_run_id", "scored_at"]:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Versioning column '{col}' has null values")

    return errors
