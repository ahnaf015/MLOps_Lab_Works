import os
import json
import argparse
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


SRC_DIR = os.path.abspath(os.path.dirname(__file__))
LAB2_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

MODELS_DIR = os.path.join(LAB2_DIR, "models")
METRICS_DIR = os.path.join(LAB2_DIR, "metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp

    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}_dt_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    data = load_breast_cancer()
    X, y = data.data, data.target

    # Same split logic as training
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1_Score": float(f1_score(y_test, y_pred)),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, f"{timestamp}_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to {metrics_path}")
