"""Data validation utilities for schema and quality checks."""

import pandas as pd
from typing import List
import logging


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> None:

    """Ensure required columns exist in the DataFrame."""

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_nulls(df: pd.DataFrame, critical_columns: List[str]) -> None:

    """Check for null values in critical columns."""

    null_counts = df[critical_columns].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            logging.warning(f"Column {col} has {count} null values")


def summarize_class_balance(df: pd.DataFrame, label_col: str) -> None:

    """Log class distribution for a binary label column."""
       
    distribution = df[label_col].value_counts(normalize=True)
    logging.info(f"Class distribution:\n{distribution}")