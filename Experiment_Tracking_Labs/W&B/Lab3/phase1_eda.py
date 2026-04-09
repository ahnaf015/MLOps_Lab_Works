"""
Phase 1: Exploratory Data Analysis + W&B Artifacts & Tables
- Loads the mushroom dataset
- Logs raw + processed data as W&B Artifacts (data versioning)
- Logs EDA visualizations as W&B Tables
"""

import os
import sys
import wandb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from app.utils import (
    load_raw_data, decode_features, prepare_data,
    PROJECT_NAME, WANDB_ENTITY, DATA_PATH, ARTIFACTS_DIR, TARGET_MAP
)


def run_eda():
    """Run full EDA pipeline and log everything to W&B."""

    run = wandb.init(
        project=PROJECT_NAME,
        entity=WANDB_ENTITY,
        name="phase1-eda",
        job_type="data-exploration",
        tags=["eda", "data-versioning"],
    )

    # ------------------------------------------------------------------
    # 1. Load & log raw data as artifact
    # ------------------------------------------------------------------
    print("[Phase 1] Loading raw data...")
    raw_df = load_raw_data()
    decoded_df = decode_features(raw_df)

    raw_artifact = wandb.Artifact("mushroom-raw-data", type="dataset")
    raw_artifact.add_file(DATA_PATH)
    run.log_artifact(raw_artifact)
    print(f"  -> Logged raw data artifact ({len(raw_df)} samples)")

    # ------------------------------------------------------------------
    # 2. Log decoded data as W&B Table (interactive exploration)
    # ------------------------------------------------------------------
    print("[Phase 1] Logging data table to W&B...")
    data_table = wandb.Table(dataframe=decoded_df)
    run.log({"dataset/full_table": data_table})

    # ------------------------------------------------------------------
    # 3. Class distribution
    # ------------------------------------------------------------------
    print("[Phase 1] Analyzing class distribution...")
    class_counts = decoded_df["class"].value_counts()
    class_table = wandb.Table(
        columns=["class", "count", "percentage"],
        data=[
            [cls, count, round(count / len(decoded_df) * 100, 1)]
            for cls, count in class_counts.items()
        ]
    )
    run.log({"eda/class_distribution": class_table})

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#e74c3c"]
    class_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Class Distribution: Edible vs Poisonous")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    run.log({"eda/class_distribution_plot": wandb.Image(fig)})
    plt.close()

    # ------------------------------------------------------------------
    # 4. Feature distributions (top discriminative features)
    # ------------------------------------------------------------------
    print("[Phase 1] Analyzing feature distributions...")
    key_features = ["odor", "gill-color", "spore-print-color", "habitat", "cap-color", "population"]

    for feat in key_features:
        if feat not in decoded_df.columns:
            continue
        cross_tab = pd.crosstab(decoded_df[feat], decoded_df["class"])
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_tab.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_title(f"{feat} vs Class")
        ax.set_ylabel("Count")
        ax.legend(title="Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        run.log({f"eda/feature_{feat}": wandb.Image(fig)})
        plt.close()

    # ------------------------------------------------------------------
    # 5. Feature-class correlation table
    # ------------------------------------------------------------------
    print("[Phase 1] Computing feature importance (chi-squared)...")
    from sklearn.feature_selection import chi2
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, feature_names = prepare_data()

    chi2_scores, p_values = chi2(X_train, y_train)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "chi2_score": np.round(chi2_scores, 2),
        "p_value": np.round(p_values, 6),
    }).sort_values("chi2_score", ascending=False)

    importance_table = wandb.Table(dataframe=importance_df)
    run.log({"eda/feature_importance_chi2": importance_table})

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["chi2_score"], color="#3498db")
    ax.set_xlabel("Chi-squared Score")
    ax.set_title("Feature Importance (Chi-squared Test)")
    ax.invert_yaxis()
    plt.tight_layout()
    run.log({"eda/feature_importance_plot": wandb.Image(fig)})
    plt.close()

    # ------------------------------------------------------------------
    # 6. Log processed data splits as artifact
    # ------------------------------------------------------------------
    print("[Phase 1] Logging processed data splits...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    processed_artifact = wandb.Artifact("mushroom-processed-data", type="dataset")

    for split_name, (X, y) in splits.items():
        split_df = pd.DataFrame(X, columns=feature_names)
        split_df["target"] = y
        path = os.path.join(ARTIFACTS_DIR, f"{split_name}.csv")
        split_df.to_csv(path, index=False)
        processed_artifact.add_file(path)

    # Save feature names and encoders info
    meta = {"feature_names": feature_names, "n_train": len(X_train),
            "n_val": len(X_val), "n_test": len(X_test)}
    processed_artifact.metadata = meta
    run.log_artifact(processed_artifact)

    # ------------------------------------------------------------------
    # 7. Dataset summary stats
    # ------------------------------------------------------------------
    summary_table = wandb.Table(
        columns=["metric", "value"],
        data=[
            ["Total samples", len(raw_df)],
            ["Features", len(feature_names)],
            ["Train samples", len(X_train)],
            ["Validation samples", len(X_val)],
            ["Test samples", len(X_test)],
            ["Edible %", round(Counter(y_train)[0] / len(y_train) * 100, 1)],
            ["Poisonous %", round(Counter(y_train)[1] / len(y_train) * 100, 1)],
        ]
    )
    run.log({"eda/dataset_summary": summary_table})

    print("[Phase 1] EDA complete!")
    run.finish()
    return True


if __name__ == "__main__":
    run_eda()
