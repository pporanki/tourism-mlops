"""
Model Building with Experiment Tracking
---------------------------------------
1. Loads train/test data from the Hugging Face dataset space.
2. Builds a preprocessing + XGBoost pipeline.
3. Tunes hyper-parameters with GridSearchCV.
4. Logs every tuned parameter and metric to MLflow.
5. Evaluates the best model on the held-out test set.
6. Registers (uploads) the best model to the Hugging Face model hub.

Run locally with:
    export HF_TOKEN=hf_xxx
    python tourism_project/model_building/train.py
"""

import os

import joblib
import mlflow
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download
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

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"

TARGET = "ProdTaken"
RANDOM_STATE = 42
LOCAL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(LOCAL_DIR, "best_tourism_model.joblib")


def load_split(filename: str, token: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    return pd.read_csv(path)


def build_pipeline(num_cols, cat_cols) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    # Handle class imbalance (~80/20) with scale_pos_weight ~ 4
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=4,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    # 1. Load train/test from HF
    train_df = load_split("train.csv", token)
    test_df = load_split("test.csv", token)
    print(f"[train] train={train_df.shape}, test={test_df.shape}")

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    pipe = build_pipeline(num_cols, cat_cols)

    # 2. Hyper-parameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }

    # 3. Experiment tracking with MLflow
    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run(run_name="xgboost-gridsearch"):
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # 4. Log all tuned parameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # 5. Evaluate on the test set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "best_cv_f1": grid.best_score_,
        }
        mlflow.log_metrics(metrics)

        print("[train] Best params:", grid.best_params_)
        print("[train] Test metrics:", metrics)
        print(classification_report(y_test, y_pred))

        # 6. Persist and register the best model on the HF model hub
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        api = HfApi(token=token)
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=token,
            private=False,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="best_tourism_model.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"[train] Registered best model to {MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
