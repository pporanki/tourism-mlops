"""
Data Preparation
----------------
1. Loads the raw dataset directly from the Hugging Face dataset space.
2. Cleans the data (fixes inconsistent categories, drops unnecessary columns,
   imputes missing values).
3. Splits into stratified train / test sets and saves them locally.
4. Uploads the resulting train.csv and test.csv back to the HF dataset space.

Run locally with:
    export HF_TOKEN=hf_xxx
    python tourism_project/data/prep.py
"""

import os

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"
REPO_TYPE = "dataset"

TARGET = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.20

LOCAL_DIR = os.path.dirname(__file__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data-cleaning rules and return a tidy frame."""
    # Drop the unnamed index column if it survived the read
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # CustomerID is a unique identifier with no predictive value
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Fix typo'd / inconsistent categories
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    # "Unmarried" and "Single" mean the same thing for this business
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Impute missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # 1. Load raw data directly from the HF dataset space
    raw_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="tourism.csv",
        repo_type=REPO_TYPE,
        token=token,
    )
    df = pd.read_csv(raw_path)
    print(f"[prep] Loaded raw data from HF: {df.shape}")

    # 2. Clean
    df = clean(df)
    print(f"[prep] Cleaned data: {df.shape}")

    # 3. Stratified split (preserves the 80/20 class balance) and save locally
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    train_path = os.path.join(LOCAL_DIR, "train.csv")
    test_path = os.path.join(LOCAL_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[prep] Saved train={train_df.shape}, test={test_df.shape}")

    # 4. Upload train / test back to the HF dataset space
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
