"""
Phase 2: Model Training + W&B Sweeps
- Trains XGBoost, LightGBM, or Random Forest based on sweep config
- Logs metrics, confusion matrix, ROC curve, feature importance
- W&B Alerts when a model beats the current best F1
"""

import os
import sys
import wandb
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from app.utils import prepare_data, PROJECT_NAME, WANDB_ENTITY, ARTIFACTS_DIR


def build_model(config):
    """Build a model based on sweep config."""
    model_type = config.model_type

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            n_estimators=config.n_estimators,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            n_estimators=config.n_estimators,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=max(2, config.min_child_weight),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def log_evaluation_metrics(run, y_true, y_pred, y_prob, feature_names, model, prefix="val"):
    """Log comprehensive evaluation metrics to W&B."""

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    run.log({
        f"{prefix}/accuracy": acc,
        f"{prefix}/f1": f1,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
    })

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Edible", "Poisonous"],
                yticklabels=["Edible", "Poisonous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({prefix})")
    plt.tight_layout()
    run.log({f"{prefix}/confusion_matrix": wandb.Image(fig)})
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({prefix})")
    ax.legend()
    plt.tight_layout()
    run.log({f"{prefix}/roc_curve": wandb.Image(fig)})
    plt.close()

    run.log({f"{prefix}/roc_auc": roc_auc})

    # Precision-Recall Curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_curve, prec_curve, color="#e74c3c", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({prefix})")
    ax.legend()
    plt.tight_layout()
    run.log({f"{prefix}/pr_curve": wandb.Image(fig)})
    plt.close()

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in idx], importances[idx], color="#2ecc71")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        run.log({f"{prefix}/feature_importance": wandb.Image(fig)})
        plt.close()

        # Also log as table
        importance_table = wandb.Table(
            columns=["feature", "importance"],
            data=[[feature_names[i], float(importances[i])] for i in idx]
        )
        run.log({f"{prefix}/feature_importance_table": importance_table})

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "roc_auc": roc_auc}


def train():
    """Single training run (called by sweep agent or directly)."""

    run = wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, job_type="training")
    config = wandb.config

    print(f"[Phase 2] Training {config.model_type} model...")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, feature_names = prepare_data()

    # Build and train
    model = build_model(config)
    model.fit(X_train, y_train)

    # Predict
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Log validation metrics
    val_metrics = log_evaluation_metrics(run, y_val, y_val_pred, y_val_prob, feature_names, model, "val")

    # Log test metrics
    test_metrics = log_evaluation_metrics(run, y_test, y_test_pred, y_test_prob, feature_names, model, "test")

    # Save model artifact — unique file per model type
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, f"model_{config.model_type}.joblib")
    joblib.dump(model, model_path)

    # Also save as "model.joblib" (default for API)
    default_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(model, default_path)

    # Also save encoders
    encoders_path = os.path.join(ARTIFACTS_DIR, "encoders.joblib")
    joblib.dump(encoders, encoders_path)

    # Save feature names
    meta_path = os.path.join(ARTIFACTS_DIR, "feature_names.joblib")
    joblib.dump(feature_names, meta_path)

    model_artifact = wandb.Artifact(
        f"mushroom-model-{config.model_type}",
        type="model",
        metadata={
            "model_type": config.model_type,
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
        }
    )
    model_artifact.add_file(model_path)
    model_artifact.add_file(encoders_path)
    model_artifact.add_file(meta_path)
    run.log_artifact(model_artifact)

    # W&B Alert if F1 is very high
    if val_metrics["f1"] > 0.98:
        wandb.alert(
            title="High-Performance Model Found!",
            text=f"{config.model_type} achieved F1={val_metrics['f1']:.4f} on validation set",
            level=wandb.AlertLevel.INFO,
        )

    print(f"[Phase 2] {config.model_type} — Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}")

    run.finish()
    return val_metrics


def run_sweep(count=10):
    """Run a W&B sweep with Bayesian optimization."""
    import yaml

    sweep_config_path = os.path.join(os.path.dirname(__file__), "sweep_config.yaml")
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Remove program key (not needed when calling from Python)
    sweep_config.pop("program", None)

    print(f"[Phase 2] Starting W&B Sweep ({count} runs, Bayesian optimization)...")
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=WANDB_ENTITY)
    wandb.agent(sweep_id, function=train, count=count)
    print(f"[Phase 2] Sweep complete! Sweep ID: {sweep_id}")
    return sweep_id


def train_single(model_type="xgboost"):
    """Train a single model without sweep (for quick testing)."""
    wandb.init(
        project=PROJECT_NAME,
        entity=WANDB_ENTITY,
        config={
            "model_type": model_type,
            "learning_rate": 0.1,
            "max_depth": 5,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )
    return train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run W&B sweep")
    parser.add_argument("--count", type=int, default=10, help="Number of sweep runs")
    parser.add_argument("--model", type=str, default="xgboost", help="Model type for single run")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(count=args.count)
    else:
        train_single(model_type=args.model)
