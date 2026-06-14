"""
Data Registration
------------------
Uploads the raw tourism.csv to a Hugging Face *dataset* repository so that every
downstream stage of the pipeline (preparation, training, deployment) can pull a
single, versioned source of truth instead of relying on a local copy.

Run locally with:
    export HF_TOKEN=hf_xxx          # (Windows PowerShell: $env:HF_TOKEN="hf_xxx")
    python tourism_project/data/data_register.py
"""

import os

from huggingface_hub import HfApi, create_repo

# --------------------------------------------------------------------------- #
# Configuration  -- replace HF_USERNAME with your own Hugging Face username    #
# --------------------------------------------------------------------------- #
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"
REPO_TYPE = "dataset"

# Path to the raw file (relative to repo root)
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "tourism.csv")


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # 1. Create the dataset repo (no-op if it already exists)
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        private=False,
        exist_ok=True,
    )
    print(f"[data_register] Dataset repo ready: {DATASET_REPO_ID}")

    # 2. Upload the raw csv
    api.upload_file(
        path_or_fileobj=LOCAL_CSV,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"[data_register] Uploaded raw tourism.csv to {DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
