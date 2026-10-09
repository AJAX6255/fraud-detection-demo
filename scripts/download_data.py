"""Download the Sparkov credit-card fraud dataset from Kaggle into data/raw/.

Requires kagglehub (anonymous downloads usually work; otherwise place the CSVs
from https://www.kaggle.com/datasets/kartik2112/fraud-detection into data/raw/ manually).

Usage: python -m scripts.download_data
"""
import os
import shutil


def main():
    import kagglehub
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    os.makedirs("data/raw", exist_ok=True)
    for f in os.listdir(path):
        if f.endswith(".csv"):
            shutil.copy(os.path.join(path, f), os.path.join("data/raw", f))
            print("saved data/raw/" + f)
    print("\nNext steps:")
    print("  pandas.concat([fraudTrain, fraudTest]) -> data/raw/all.csv")
    print("  python -m scripts.train_model --data data/raw/all.csv")
    print("  python -m scripts.prepare_app_data --data data/raw/all.csv")


if __name__ == "__main__":
    main()
