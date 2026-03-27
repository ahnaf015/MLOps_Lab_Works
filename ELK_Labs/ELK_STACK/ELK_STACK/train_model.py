"""
Network Intrusion Detection - ML Training Pipeline
====================================================
Trains multiple classifiers on the KDD Cup 99 dataset for network
intrusion detection and outputs structured JSON logs for ingestion
by the ELK Stack (Elasticsearch, Logstash, Kibana).

Dataset : KDD Cup 1999 (auto-downloaded via scikit-learn)
Task    : Binary classification — Normal traffic vs Attack traffic
Models  : Logistic Regression, Random Forest, Gradient Boosting

Usage:
    pip install -r requirements.txt
    python train_model.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.log")
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000   # Set to None to use the full 10% subset (~494k rows)

# KDD Cup 99 column names (41 features)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERIC_COLS = [c for c in COLUMN_NAMES if c not in CATEGORICAL_COLS]


# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object (JSON Lines)."""

    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event_type": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, default=str)


def setup_logging():
    """Configure dual logging: JSON Lines to file, human-readable to console."""
    logger = logging.getLogger("training_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler — structured JSON lines for Logstash
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console handler — human-readable for terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(logger):
    """Load KDD Cup 99 dataset and prepare for binary classification."""

    logger.info("pipeline_start", extra={"data": {
        "pipeline": "network_intrusion_detection",
        "dataset": "KDD Cup 99 (10% subset)",
        "task": "binary_classification",
        "sample_size": SAMPLE_SIZE,
        "test_size": TEST_SIZE,
    }})

    # -- Fetch dataset (auto-downloads on first run) -------------------------
    logger.info("data_loading", extra={"data": {
        "source": "sklearn.datasets.fetch_kddcup99",
        "percent10": True,
    }})
    kdd = fetch_kddcup99(subset=None, percent10=True, random_state=RANDOM_STATE)

    # -- Build DataFrame -----------------------------------------------------
    df = pd.DataFrame(kdd.data, columns=COLUMN_NAMES)

    # Decode bytes columns produced by sklearn
    for col in CATEGORICAL_COLS:
        df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Target: binary (normal = 0, any attack = 1)
    raw_labels = np.array([
        lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in kdd.target
    ])
    df["label"] = raw_labels
    df["is_attack"] = (df["label"] != "normal.").astype(int)

    # -- Optional: subsample for faster training -----------------------------
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    # -- Log dataset profile -------------------------------------------------
    n_total = len(df)
    n_normal = int((df["is_attack"] == 0).sum())
    n_attack = int((df["is_attack"] == 1).sum())

    top_attacks = (
        df.loc[df["is_attack"] == 1, "label"]
        .value_counts()
        .head(10)
        .to_dict()
    )
    protocol_dist = df["protocol_type"].value_counts().to_dict()

    logger.info("data_profile", extra={"data": {
        "total_samples": n_total,
        "normal_traffic": n_normal,
        "attack_traffic": n_attack,
        "attack_ratio": round(n_attack / n_total, 4),
        "num_features": len(COLUMN_NAMES),
        "categorical_features": CATEGORICAL_COLS,
        "top_attack_types": {k: int(v) for k, v in top_attacks.items()},
        "protocol_distribution": protocol_dist,
    }})

    # -- Separate features / target ------------------------------------------
    X = df[COLUMN_NAMES].copy()
    y = df["is_attack"].copy()

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # -- Preprocessing pipeline ----------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )

    # -- Train / Test split --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    logger.info("data_split", extra={"data": {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_attack_ratio": round(float(y_train.mean()), 4),
        "test_attack_ratio": round(float(y_test.mean()), 4),
    }})

    return X_train, X_test, y_train, y_test, preprocessor


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def get_models():
    """Return a dict of classifiers to train and compare."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, solver="saga", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(logger, model_name, pipeline, X_test, y_test, train_time):
    """Compute and log all metrics for a trained model."""

    y_pred = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 6),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 6),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6) if y_proba is not None else None,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": round(float(fp / (fp + tn)), 6) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": round(float(fn / (fn + tp)), 6) if (fn + tp) > 0 else 0.0,
        "true_positive_rate": round(float(tp / (tp + fn)), 6) if (tp + fn) > 0 else 0.0,
        "true_negative_rate": round(float(tn / (tn + fp)), 6) if (tn + fp) > 0 else 0.0,
        "training_time_seconds": round(train_time, 4),
        "test_samples": len(y_test),
    }

    logger.info("model_evaluation", extra={"data": metrics})
    return metrics


def log_feature_importance(logger, model_name, pipeline, feature_names):
    """Log top 10 feature importances for tree-based models."""

    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return

    # Get transformed feature names
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        cat_features = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
        )
        all_features = NUMERIC_COLS + cat_features
    except Exception:
        all_features = [f"feature_{i}" for i in range(len(classifier.feature_importances_))]

    importances = classifier.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]

    top_features = [
        {"feature": all_features[i], "importance": round(float(importances[i]), 6)}
        for i in top_idx
    ]

    logger.info("feature_importance", extra={"data": {
        "model_name": model_name,
        "top_features": top_features,
    }})


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_and_evaluate(logger, X_train, X_test, y_train, y_test, preprocessor):
    """Train all models, evaluate, and log a comparison summary."""

    models = get_models()
    results = []

    for name, clf in models.items():
        logger.info("training_start", extra={"data": {"model_name": name}})

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        start = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - start

        logger.info("training_complete", extra={"data": {
            "model_name": name,
            "training_time_seconds": round(train_time, 4),
        }})

        metrics = evaluate_model(logger, name, pipe, X_test, y_test, train_time)
        log_feature_importance(logger, name, pipe, COLUMN_NAMES)
        results.append(metrics)

    # -- Comparison summary --------------------------------------------------
    best = max(results, key=lambda r: r["f1_score"])

    logger.info("comparison_summary", extra={"data": {
        "models_trained": len(results),
        "best_model": best["model_name"],
        "best_f1_score": best["f1_score"],
        "best_accuracy": best["accuracy"],
        "best_roc_auc": best["roc_auc"],
        "all_results": [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "f1_score": r["f1_score"],
                "roc_auc": r["roc_auc"],
                "train_time_s": r["training_time_seconds"],
            }
            for r in results
        ],
    }})

    logger.info("pipeline_complete", extra={"data": {
        "status": "success",
        "total_models": len(results),
        "best_model": best["model_name"],
    }})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()

    try:
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(logger)
        results = train_and_evaluate(
            logger, X_train, X_test, y_train, y_test, preprocessor,
        )

        # Pretty-print summary table to console
        print("\n" + "=" * 75)
        print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'Time(s)':>10}")
        print("-" * 75)
        for r in results:
            print(
                f"  {r['model_name']:<25}"
                f" {r['accuracy']:>10.4f}"
                f" {r['f1_score']:>10.4f}"
                f" {r['roc_auc']:>10.4f}"
                f" {r['training_time_seconds']:>10.2f}"
            )
        print("=" * 75)

        best = max(results, key=lambda r: r["f1_score"])
        print(f"\n  Best model: {best['model_name']} (F1={best['f1_score']:.4f})\n")

    except Exception as e:
        logger.error("pipeline_error", extra={"data": {
            "error": str(e),
            "type": type(e).__name__,
        }})
        raise


if __name__ == "__main__":
    main()
