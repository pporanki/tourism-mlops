"""
============================================================================
STAGE 3 of 4 - MODEL BUILDING (with experiment tracking)
============================================================================
Purpose
-------
  1. Download the train/test splits from the Hugging Face dataset space.
  2. Build a single scikit-learn Pipeline = preprocessing + XGBoost classifier.
  3. Tune hyper-parameters with GridSearchCV (cross-validated, scored on F1).
  4. Log every chosen parameter and test metric to MLflow for reproducibility.
  5. Evaluate the best model on the held-out test set.
  6. Register (upload) the trained model to the Hugging Face MODEL hub so the
     deployment stage / Streamlit app can load it.

Where it runs
-------------
* Locally  : `python tourism_project/model_building/train.py`
* In CI    : the third job (`model-training`), after data-preparation.
============================================================================
"""

import os

import joblib            # serialise the trained pipeline to a .joblib file
import mlflow            # experiment tracking (params + metrics + artifacts)
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, hf_hub_download

# Preprocessing + modelling building blocks
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Load credentials from .env locally (no-op in CI / GitHub Secrets).
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"                  # where splits live
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"      # where the model goes

TARGET = "ProdTaken"     # label column (1 = purchased)
RANDOM_STATE = 42        # reproducible model + CV folds
LOCAL_DIR = os.path.dirname(__file__)
# Local filename for the serialised pipeline before it is uploaded to the Hub.
MODEL_PATH = os.path.join(LOCAL_DIR, "best_tourism_model.joblib")


def load_split(filename: str, token: str) -> pd.DataFrame:
    """Download one split (train.csv / test.csv) from the HF dataset space."""
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    return pd.read_csv(path)


def build_pipeline(num_cols, cat_cols) -> Pipeline:
    """
    Build the full preprocessing + model Pipeline.

    Bundling preprocessing and the model into ONE object guarantees the exact
    same transformations are applied at training time and at inference time
    (no train/serve skew) and lets us tune everything together.
    """
    # ColumnTransformer applies a different transformer to each group of columns:
    #   - numeric     -> StandardScaler (zero mean, unit variance)
    #   - categorical -> OneHotEncoder (one 0/1 column per category)
    # handle_unknown="ignore" means categories unseen during training are
    # encoded as all-zeros at inference instead of raising an error.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # XGBoost classifier. scale_pos_weight ~ 4 tells the model the negative
    # class is ~4x more common, counteracting the ~80/20 imbalance so the
    # minority (purchasers) is not ignored.
    model = XGBClassifier(
        objective="binary:logistic",   # binary classification with probabilities
        eval_metric="logloss",         # training loss metric
        random_state=RANDOM_STATE,
        scale_pos_weight=4,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    """Train, tune, evaluate, track, and register the model."""

    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    # 1. Load the train/test splits produced by the data-preparation stage.
    train_df = load_split("train.csv", token)
    test_df = load_split("test.csv", token)
    print(f"[train] train={train_df.shape}, test={test_df.shape}")

    # Separate features (X) from the label (y).
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    # Detect which columns are numeric vs categorical so each gets the right
    # preprocessing inside the ColumnTransformer.
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    pipe = build_pipeline(num_cols, cat_cols)

    # 2. Hyper-parameter search space. The "model__" prefix routes each value to
    #    the "model" step inside the pipeline. This grid = 2*3*2*2 = 24 combos.
    param_grid = {
        "model__n_estimators": [100, 200],     # number of boosting trees
        "model__max_depth": [3, 5, 7],         # tree depth (model complexity)
        "model__learning_rate": [0.05, 0.1],   # shrinkage per tree
        "model__subsample": [0.8, 1.0],        # row sampling per tree
    }

    # 3. Track this experiment with MLflow. set_experiment groups runs; start_run
    #    opens a single tracked run that we log params/metrics/artifacts into.
    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run(run_name="xgboost-gridsearch"):
        # GridSearchCV tries every combo with 5-fold cross-validation and keeps
        # the one with the best F1. n_jobs=-1 uses all CPU cores in parallel.
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1",   # F1 balances precision/recall on the minority class
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_   # pipeline refit with the best params

        # 4. Log the winning hyper-parameters and search settings to MLflow.
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # 5. Evaluate on the held-out test set.
        y_pred = best_model.predict(X_test)                 # 0/1 class labels
        y_proba = best_model.predict_proba(X_test)[:, 1]    # P(purchase) for ROC-AUC
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "best_cv_f1": grid.best_score_,
        }
        mlflow.log_metrics(metrics)   # record metrics in the MLflow run

        print("[train] Best params:", grid.best_params_)
        print("[train] Test metrics:", metrics)
        print(classification_report(y_test, y_pred))

        # 6. Persist the trained pipeline to disk, attach it to the MLflow run,
        #    then register it on the Hugging Face MODEL hub.
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)   # keep the model alongside its metrics

        api = HfApi(token=token)
        # Create the model repo (idempotent) ...
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=token,
            private=False,
            exist_ok=True,
        )
        # ... and upload the serialised pipeline so the app can download it.
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="best_tourism_model.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"[train] Registered best model to {MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
