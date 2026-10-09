"""CLI entry point for training.

Usage (from repo root):
  python -m scripts.train_model --data sample                          # synthetic smoke test
  python -m scripts.train_model --data data/raw/fraudTrain.csv         # Sparkov / Kaggle
  python -m scripts.train_model --data data/raw/all.csv --out artifacts
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/parquet path, or 'sample' to generate first")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    if a.data == "sample":
        from scripts.make_sample_data import generate
        a.data = generate()

    df = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
    print(f"loaded {len(df):,} rows from {a.data} (fraud rate {df['is_fraud'].mean():.4f})")

    from src.train import train
    metrics = train(df, a.out)
    print(json.dumps(metrics, indent=2))
    print(f"artifacts written to {a.out}/")


if __name__ == "__main__":
    main()
