"""
============================================================================
STAGE 2 of 4 - DATA PREPARATION
============================================================================
Purpose
-------
Turn the raw dataset into clean, model-ready train/test splits:
  1. Download the raw `tourism.csv` from the Hugging Face dataset space.
  2. Clean it (fix inconsistent categories, drop useless columns, impute NaNs).
  3. Split into stratified train / test sets (preserving the class balance).
  4. Upload `train.csv` and `test.csv` back to the same HF dataset space so the
     training stage downloads the EXACT same splits.

Where it runs
-------------
* Locally  : `python tourism_project/data/prep.py`
* In CI    : the second job (`data-preparation`), after data-registration.
============================================================================
"""

import os

import pandas as pd
from dotenv import load_dotenv

# hf_hub_download -> download a single file from the Hub and return its local path
# HfApi          -> client used to upload the resulting splits back to the Hub
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# Load HF_USERNAME / HF_TOKEN from .env locally (no-op in CI / GitHub Secrets).
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
# `or` (not a getenv default) so an EMPTY value -- e.g. an unset GitHub secret
# that expands to "" -- also falls back instead of producing "/tourism".
HF_USERNAME = os.getenv("HF_USERNAME") or "prudvikrishna"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"   # same dataset repo as stage 1
REPO_TYPE = "dataset"

TARGET = "ProdTaken"     # the label column (1 = customer purchased the package)
RANDOM_STATE = 42        # fixed seed -> identical split on every run
TEST_SIZE = 0.20         # hold out 20% of rows for the test set

# Folder this script lives in; train.csv / test.csv are written here.
LOCAL_DIR = os.path.dirname(__file__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every data-cleaning rule and return a tidy copy.

    The rules below come directly from the EDA findings in the notebook:
    inconsistent category spellings, a non-predictive id column, and a few
    missing values.
    """
    # Drop any stray "Unnamed: 0" index column that a previous CSV save may have
    # left behind (the regex matches column names starting with "Unnamed").
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # CustomerID is a unique identifier -> zero predictive value, so remove it
    # to avoid the model "memorising" individual customers.
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Fix the data-entry typo "Fe Male" so Gender has exactly two clean values.
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # "Unmarried" and "Single" mean the same thing for this business -> merge
    # them into one category so the model does not treat them as different.
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Impute any remaining missing values, column by column:
    #   - categorical (object) columns -> fill with the most frequent value (mode)
    #   - numeric columns              -> fill with the median (robust to outliers)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def main() -> None:
    """Download raw data, clean, split, save, and upload the splits to the Hub."""

    # Token is required to download from a (possibly private) repo and to upload.
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # 1. Download the raw CSV from the HF dataset space. hf_hub_download caches
    #    the file locally and returns the path to it.
    raw_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="tourism.csv",
        repo_type=REPO_TYPE,
        token=token,
    )
    df = pd.read_csv(raw_path)
    print(f"[prep] Loaded raw data from HF: {df.shape}")

    # 2. Clean the raw data with the rules defined above.
    df = clean(df)
    print(f"[prep] Cleaned data: {df.shape}")

    # 3. Stratified train/test split.
    #    stratify=df[TARGET] -> the ~19% purchase rate is preserved in BOTH
    #    splits, which matters a lot for an imbalanced classification problem.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    # Save the splits next to this script. index=False -> do not write the
    # pandas row index as an extra column.
    train_path = os.path.join(LOCAL_DIR, "train.csv")
    test_path = os.path.join(LOCAL_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[prep] Saved train={train_df.shape}, test={test_df.shape}")

    # 4. Upload both splits back to the dataset repo so training reads the same
    #    data. These files are git-ignored locally; the Hub is their home.
    for path, name in [(train_path, "train.csv"), (test_path, "test.csv")]:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=DATASET_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"[prep] Uploaded {name} to {DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
