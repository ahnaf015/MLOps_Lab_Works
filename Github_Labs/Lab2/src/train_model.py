import os
import sys
import argparse
import datetime
import pickle

import mlflow
from joblib import dump

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


SRC_DIR = os.path.abspath(os.path.dirname(__file__))
LAB2_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

DATA_DIR = os.path.join(SRC_DIR, "data")
MLRUNS_DIR = os.path.join(SRC_DIR, "mlruns")
MODELS_DIR = os.path.join(LAB2_DIR, "models")


def to_file_uri(path: str) -> str:
    # Convert Windows path to file URI that MLflow understands
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in [i / 100 for i in range(5, 96)]:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Save dataset artifacts into src/data
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "data.pickle"), "wb") as f:
        pickle.dump(X, f)
    with open(os.path.join(DATA_DIR, "target.pickle"), "wb") as f:
        pickle.dump(y, f)

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)),
        ]
    )

    # MLflow setup (save into src/mlruns)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(to_file_uri(MLRUNS_DIR))

    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        mlflow.log_params(
            {
                "dataset_name": dataset_name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "model": "LogisticRegression + StandardScaler",
                "split_train": len(y_train),
                "split_val": len(y_val),
                "split_test": len(y_test),
            }
        )

        model.fit(X_train, y_train)

        # Threshold tuning on VAL
        val_probs = model.predict_proba(X_val)[:, 1]
        best_t, best_val_f1 = find_best_threshold(y_val, val_probs)
        mlflow.log_params({"best_threshold": best_t})
        mlflow.log_metrics({"val_f1_at_best_threshold": best_val_f1})

        # Test metrics using tuned threshold
        test_probs = model.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= best_t).astype(int)
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_f1": f1_score(y_test, test_pred),
            }
        )

        # Save model directly into Lab2/models with workflow expected name
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_filename = os.path.join(MODELS_DIR, f"model_{timestamp}_dt_model.joblib")
        dump(model, model_filename)
        mlflow.log_artifact(model_filename)

    print(f"Saved model to: {model_filename}")
