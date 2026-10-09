"""Build the parquet the Streamlit app reads: keep ALL fraud rows + a sample of
legit ones, so the live feed surfaces fraud regularly without a huge file.

Usage: python -m scripts.prepare_app_data --data data/sample_transactions.csv
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/processed/transactions.parquet")
    ap.add_argument("--max-legit", type=int, default=150_000)
    a = ap.parse_args()

    df = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
    fraud = df[df["is_fraud"] == 1]
    legit_pool = df[df["is_fraud"] == 0]
    legit = legit_pool.sample(min(a.max_legit, len(legit_pool)), random_state=42)
    out = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out):,} rows ({len(fraud):,} fraud) -> {a.out}")


if __name__ == "__main__":
    main()
