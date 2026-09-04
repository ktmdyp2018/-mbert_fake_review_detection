"""
Downloads the two datasets referenced in the paper (Section III.A):

  [31] Amazon English review dataset (Kaggle) - requires a Kaggle account
       and API token (kaggle.json), since Kaggle needs authentication and
       cannot be fetched anonymously.
  [32] Amazon Hindi Fake Review Dataset (GitHub) - fetched automatically
       below, since it is hosted as a public raw file.

Usage:
    python scripts/download_datasets.py
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

HINDI_XLSX_URL = (
    "https://raw.githubusercontent.com/ktmdyp2018/hindireviewdatasetofamazon/"
    "main/Hindi_Fake_Review_Dataset.xlsx"
)

KAGGLE_DATASET_SLUG = "nehaprabhavalkar/analysis-of-indian-product-reviews-on-amazon"


def download_hindi_dataset():
    os.makedirs(PATHS.data_dir, exist_ok=True)
    dest = PATHS.hindi_xlsx
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists.")
        return
    print(f"Downloading Hindi dataset from {HINDI_XLSX_URL} ...")
    try:
        urllib.request.urlretrieve(HINDI_XLSX_URL, dest)
        print(f"[ok] saved to {dest}")
    except Exception as e:
        print(f"[error] could not download Hindi dataset automatically: {e}")
        print("Please download it manually from:")
        print("  https://github.com/ktmdyp2018/hindireviewdatasetofamazon")
        print(f"and place the .xlsx file at: {dest}")


def print_kaggle_instructions():
    dest = PATHS.english_csv
    print()
    print("=" * 78)
    print("English dataset (Kaggle) requires authentication and cannot be")
    print("fetched automatically. To download it:")
    print()
    print("  1. pip install kaggle")
    print("  2. Place your Kaggle API token at ~/.kaggle/kaggle.json")
    print(f"  3. kaggle datasets download -d {KAGGLE_DATASET_SLUG}")
    print("  4. unzip the downloaded archive")
    print(f"  5. rename/move the resulting review CSV to: {dest}")
    print()
    print("Expected columns in the CSV (see src/dataset.py / src/labeling.py):")
    print("  product_id, reviewer_id, review_text, rating, timestamp,")
    print("  account_age_days, verified_purchase")
    print("=" * 78)


if __name__ == "__main__":
    download_hindi_dataset()
    print_kaggle_instructions()
