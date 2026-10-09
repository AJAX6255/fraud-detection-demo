"""Synchronize artifacts and dataset from Google Cloud Storage on startup."""
import os
from pathlib import Path


def download_artifact(bucket_name: str, source_blob: str, dest_path: str):
    target = Path(dest_path)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(source_blob)
            if blob.exists():
                print(f"==> Downloading gs://{bucket_name}/{source_blob} -> {dest_path}...")
                blob.download_to_filename(str(target))
                print(f"==> Downloaded {dest_path}")
        except Exception as e:
            print(f"==> Warning: could not download {source_blob} from GCS: {e}")


def sync_all_from_gcs():
    bucket = os.environ.get("GCS_BUCKET")
    if not bucket:
        return
    print(f"==> Syncing artifacts with GCS bucket: {bucket}")
    download_artifact(bucket, "artifacts/model.joblib", "artifacts/model.joblib")
    download_artifact(bucket, "artifacts/shap_explainer.joblib", "artifacts/shap_explainer.joblib")
    download_artifact(bucket, "artifacts/metrics.json", "artifacts/metrics.json")
    download_artifact(bucket, "artifacts/shap_summary.png", "artifacts/shap_summary.png")
    download_artifact(bucket, "data/processed/transactions.parquet", "data/processed/transactions.parquet")
