"""Utility functions for interacting with Google Cloud Storage."""

import gcsfs
import pandas as pd


def list_parquet_files(bucket: str, prefix: str) -> list[str]:
    """List parquet file paths in a GCS bucket under a given prefix."""
    fs = gcsfs.GCSFileSystem()
    files = fs.ls(f"{bucket}/{prefix}")
    return [f"gs://{f}" for f in files if f.endswith(".parquet")]


def read_parquet_from_gcs(paths: list[str]) -> pd.DataFrame:
    """Read multiple parquet files from GCS into a single DataFrame."""
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def write_parquet_to_gcs(df: pd.DataFrame, bucket: str, path: str) -> None:
    """Write a DataFrame as parquet to a GCS bucket."""
    full_path = f"gs://{bucket}/{path}"
    df.to_parquet(full_path, index=False)
