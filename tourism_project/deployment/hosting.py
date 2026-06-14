"""
============================================================================
STAGE 4 of 4 - DEPLOYMENT (host the Streamlit app on a HF Space)
============================================================================
Purpose
-------
Create (if needed) a Hugging Face *Space* that uses the Docker SDK, then push
the three deployment files into it:
    app.py            -> the Streamlit front-end
    Dockerfile        -> how the Space builds its container
    requirements.txt  -> the app's Python dependencies
Once these files land in the Space, Hugging Face automatically builds the Docker
image and serves the Streamlit app at a public URL.

Where it runs
-------------
* Locally  : `python tourism_project/deployment/hosting.py`
* In CI    : the fourth/last job (`deployment`), after model-training.
============================================================================
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load credentials from .env locally (no-op in CI / GitHub Secrets).
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
# Full id of the Space repo, e.g. "prudvikrishna/tourism-package-predictor".
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-predictor"
REPO_TYPE = "space"

# Folder this script lives in (the deployment folder that holds the files below).
DEPLOY_DIR = os.path.dirname(__file__)
# The files the Space needs to build and run the app.
FILES_TO_PUSH = ["app.py", "Dockerfile", "requirements.txt"]


def main() -> None:
    """Create the Space and upload the deployment files into it."""

    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # 1. Create the Space.
    #    space_sdk="docker" -> the Space builds from our Dockerfile (gives us
    #                          full control over the runtime environment).
    #    exist_ok=True      -> safe to re-run; does nothing if it already exists.
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        private=False,
        exist_ok=True,
        space_sdk="docker",
    )
    print(f"[hosting] Space ready: {SPACE_REPO_ID}")

    # 2. Upload each deployment file into the Space. Pushing a file triggers the
    #    Space to (re)build its Docker image and restart the app.
    for fname in FILES_TO_PUSH:
        api.upload_file(
            path_or_fileobj=os.path.join(DEPLOY_DIR, fname),
            path_in_repo=fname,
            repo_id=SPACE_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"[hosting] Pushed {fname} to {SPACE_REPO_ID}")

    print(f"[hosting] Done. Visit https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
