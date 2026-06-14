"""
Hosting script
--------------
Creates (if needed) a Hugging Face *Space* that uses the Docker SDK and pushes
all deployment files (app.py, Dockerfile, requirements.txt) into it. Once the
files land, the Space builds the image and serves the Streamlit front-end.

Run locally with:
    export HF_TOKEN=hf_xxx
    python tourism_project/deployment/hosting.py
"""

import os

from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-predictor"
REPO_TYPE = "space"

DEPLOY_DIR = os.path.dirname(__file__)
FILES_TO_PUSH = ["app.py", "Dockerfile", "requirements.txt"]


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # 1. Create the Space with the Docker SDK (no-op if it already exists)
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        private=False,
        exist_ok=True,
        space_sdk="docker",
    )
    print(f"[hosting] Space ready: {SPACE_REPO_ID}")

    # 2. Push each deployment file into the Space
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
