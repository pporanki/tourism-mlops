"""
============================================================================
STAGE 1 of 4 - DATA REGISTRATION
============================================================================
Purpose
-------
Upload the raw `tourism.csv` to a Hugging Face *dataset* repository. From this
point on, every later stage (preparation, training, deployment) reads its data
from that single, versioned location instead of a local file. This is what
makes the pipeline reproducible: there is exactly ONE source of truth and every
machine / CI run pulls the same bytes.

Where it runs
-------------
* Locally  : `python tourism_project/data/data_register.py`
* In CI    : the first job (`data-registration`) of .github/workflows/pipeline.yml

Credentials
-----------
Reads two environment variables:
    HF_USERNAME -> your Hugging Face username (builds the repo id)
    HF_TOKEN    -> a Hugging Face WRITE token (needed to create/upload)
Locally these come from the .env file; in CI they come from GitHub Secrets.
============================================================================
"""

import os

# load_dotenv() reads key=value pairs from a local .env file into os.environ.
# It is a no-op in GitHub Actions (no .env there) where the variables are
# already injected from repository Secrets.
from dotenv import load_dotenv

# HfApi      -> client object used to talk to the Hugging Face Hub
# create_repo-> helper that creates a dataset/model/space repo on the Hub
from huggingface_hub import HfApi, create_repo

# Pull HF_USERNAME / HF_TOKEN from .env (if present) before we read them below.
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
# Username drives the repo id. The second argument to getenv is a fallback
# used only if the variable is not set anywhere.
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")

# Full id of the dataset repo on the Hub, e.g. "prudvikrishna/tourism".
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"

# We are creating a *dataset* repo (as opposed to "model" or "space").
REPO_TYPE = "dataset"

# Absolute path to the raw CSV that sits next to this script. Using
# os.path.dirname(__file__) makes the path work no matter what folder the
# script is launched from (important in CI).
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "tourism.csv")


def main() -> None:
    """Create the dataset repo (if needed) and upload the raw CSV to it."""

    # 1. Fetch the write token. Fail fast with a clear message if it is missing,
    #    because every Hub call below needs it.
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Add it to your .env file (local) or GitHub Secrets (CI)."
        )

    # 2. Build an authenticated Hub client. All uploads go through this object.
    api = HfApi(token=token)

    # 3. Create the dataset repository on the Hub.
    #    exist_ok=True  -> do nothing (no error) if the repo already exists,
    #                      so re-running the pipeline is safe and idempotent.
    #    private=False  -> the dataset is public.
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        private=False,
        exist_ok=True,
    )
    print(f"[data_register] Dataset repo ready: {DATASET_REPO_ID}")

    # 4. Upload the raw CSV into the repo.
    #    path_or_fileobj -> the local file we are sending
    #    path_in_repo    -> the name it will have inside the Hub repo
    api.upload_file(
        path_or_fileobj=LOCAL_CSV,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"[data_register] Uploaded raw tourism.csv to {DATASET_REPO_ID}")


# Standard Python entry-point guard: run main() only when this file is executed
# directly (e.g. by CI), not when it is imported by another module.
if __name__ == "__main__":
    main()
