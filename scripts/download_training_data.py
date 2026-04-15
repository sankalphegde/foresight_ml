"""Download training data from GCS to local artifacts/splits/ directory."""

from google.cloud import storage
from pathlib import Path

Path("artifacts/splits").mkdir(parents=True, exist_ok=True)

client = storage.Client()
bucket = client.bucket("financial-distress-data")

files = {
    "splits/v1/train.parquet": "artifacts/splits/train.parquet",
    "splits/v1/val.parquet": "artifacts/splits/val.parquet",
    "splits/v1/test.parquet": "artifacts/splits/test.parquet",
    "splits/v1/scaler_pipeline.pkl": "artifacts/splits/scaler_pipeline.pkl",
    "splits/v1/scale_pos_weight.json": "artifacts/splits/scale_pos_weight.json",
}

for gcs_path, local_path in files.items():
    print(f"Downloading {gcs_path} → {local_path} ...")
    bucket.blob(gcs_path).download_to_filename(local_path)

print("Done! All training data downloaded.")
