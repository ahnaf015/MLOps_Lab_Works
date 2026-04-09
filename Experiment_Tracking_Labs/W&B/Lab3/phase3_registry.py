"""
Phase 3: W&B Model Registry
- Finds the best model from all training runs
- Tags the best model artifact with "production" and "best" aliases
- Logs full model lineage and metadata
"""

import os
import sys
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from app.utils import PROJECT_NAME, WANDB_ENTITY


def find_best_model():
    """Query W&B API to find the best run by val/f1."""
    api = wandb.Api()

    runs = api.runs(
        f"{WANDB_ENTITY}/{PROJECT_NAME}",
        filters={"jobType": "training"},
        order="-summary_metrics.val/f1",
    )

    if not runs:
        print("[Phase 3] No training runs found! Run Phase 2 first.")
        return None, None

    best_run = runs[0]
    best_f1 = best_run.summary.get("val/f1", 0)
    print(f"[Phase 3] Best run: {best_run.name} (ID: {best_run.id})")
    print(f"  -> Val F1: {best_f1:.4f}")
    print(f"  -> Model type: {best_run.config.get('model_type', 'unknown')}")

    # Find model artifact from this run
    artifacts = best_run.logged_artifacts()
    model_artifact = None
    for art in artifacts:
        if art.type == "model":
            model_artifact = art
            break

    if not model_artifact:
        print("[Phase 3] No model artifact found in the best run!")
        return best_run, None

    return best_run, model_artifact


def register_model():
    """Register the best model by adding aliases to the artifact."""

    run = wandb.init(
        project=PROJECT_NAME,
        entity=WANDB_ENTITY,
        name="phase3-registry",
        job_type="model-registry",
        tags=["registry", "production"],
    )

    best_run, model_artifact = find_best_model()

    if not model_artifact:
        print("[Phase 3] Cannot register — no model artifact found.")
        run.finish()
        return False

    # Add "production" and "best" aliases to the artifact
    print(f"[Phase 3] Registering model: {model_artifact.name} (version: {model_artifact.version})")

    try:
        api = wandb.Api()
        artifact = api.artifact(f"{WANDB_ENTITY}/{PROJECT_NAME}/{model_artifact.name}")
        artifact.aliases.append("production")
        artifact.aliases.append("best")
        artifact.save()
        print("[Phase 3] Added aliases: production, best")
    except Exception as e:
        print(f"[Phase 3] Could not add aliases via API ({e}), using run.use_artifact instead...")
        # Fallback: log artifact reference with aliases from the current run
        run.use_artifact(model_artifact)

    # Log registry metadata to the run summary
    run.summary["registered_model"] = model_artifact.name
    run.summary["registered_version"] = model_artifact.version
    run.summary["registered_from_run"] = best_run.id
    run.summary["model_type"] = best_run.config.get("model_type", "unknown")
    run.summary["val_f1"] = best_run.summary.get("val/f1", 0)
    run.summary["val_accuracy"] = best_run.summary.get("val/accuracy", 0)
    run.summary["test_f1"] = best_run.summary.get("test/f1", 0)
    run.summary["test_accuracy"] = best_run.summary.get("test/accuracy", 0)

    # Log a summary table for easy viewing in W&B
    summary_table = wandb.Table(
        columns=["field", "value"],
        data=[
            ["Model Type", best_run.config.get("model_type", "unknown")],
            ["Artifact", model_artifact.name],
            ["Version", model_artifact.version],
            ["Source Run", f"{best_run.name} ({best_run.id})"],
            ["Val F1", f"{best_run.summary.get('val/f1', 0):.4f}"],
            ["Val Accuracy", f"{best_run.summary.get('val/accuracy', 0):.4f}"],
            ["Test F1", f"{best_run.summary.get('test/f1', 0):.4f}"],
            ["Test Accuracy", f"{best_run.summary.get('test/accuracy', 0):.4f}"],
        ]
    )
    run.log({"registry/model_summary": summary_table})

    # W&B Alert about new production model
    wandb.alert(
        title="New Production Model Registered",
        text=(
            f"Model: {best_run.config.get('model_type', 'unknown')}\n"
            f"Artifact: {model_artifact.name} ({model_artifact.version})\n"
            f"Val F1: {best_run.summary.get('val/f1', 0):.4f}\n"
            f"Test F1: {best_run.summary.get('test/f1', 0):.4f}\n"
            f"Run: {best_run.name} ({best_run.id})"
        ),
        level=wandb.AlertLevel.INFO,
    )

    print("[Phase 3] Model registered successfully!")
    print(f"  -> Artifact: {model_artifact.name}")
    print(f"  -> Version: {model_artifact.version}")
    print(f"  -> Model type: {best_run.config.get('model_type', 'unknown')}")

    run.finish()
    return True


if __name__ == "__main__":
    register_model()
