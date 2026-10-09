"""Combine fraudTrain.csv and fraudTest.csv into data/raw/all.parquet."""
import os
import pandas as pd


def main():
    print("==> Reading raw CSVs...")
    train_path = "data/raw/fraudTrain.csv"
    test_path = "data/raw/fraudTest.csv"
    out_parquet = "data/raw/all.parquet"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Loaded {len(df_train):,} train rows and {len(df_test):,} test rows.")
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    if "Unnamed: 0" in df_all.columns:
        df_all = df_all.drop(columns=["Unnamed: 0"])

    print(f"Total rows: {len(df_all):,} | Fraud rows: {int(df_all['is_fraud'].sum()):,} ({df_all['is_fraud'].mean():.4%})")

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df_all.to_parquet(out_parquet, index=False)
    print(f"==> Saved combined dataset to {out_parquet}")


if __name__ == "__main__":
    main()
