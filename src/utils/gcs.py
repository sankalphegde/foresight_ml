from google.cloud import storage
import pandas as pd
import gcsfs
from typing import List


def list_parquet_files(bucket: str, prefix: str) -> List[str]:
    fs = gcsfs.GCSFileSystem()
    files = fs.ls(f"{bucket}/{prefix}")
    return [f"gs://{f}" for f in files if f.endswith(".parquet")]


def read_parquet_from_gcs(paths: List[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def write_parquet_to_gcs(df: pd.DataFrame, bucket: str, path: str) -> None:
    full_path = f"gs://{bucket}/{path}"
    df.to_parquet(full_path, index=False)