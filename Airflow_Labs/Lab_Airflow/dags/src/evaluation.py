"""
evaluation.py
-------------
Generates:
  1. metrics.json — accuracy, F1, precision, recall for both models + clustering stats
  2. evaluation_plots.png — confusion matrix + ROC curve side by side
  3. feature_importance.png — top-15 feature importances (RF only, skipped for LR)

All plots use the non-interactive Agg backend so they render inside Docker.
"""

import os
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

WORKING_DATA = '/opt/airflow/working_data'


# ─── Task 1: Generate Metrics JSON ───────────────────────────────────────────

def generate_metrics(**context):
    """
    Collate metrics from RF, LR, and clustering into a single JSON file
    that the Flask dashboard can serve at /metrics.
    """
    rf_path      = os.path.join(WORKING_DATA, 'rf_model.pkl')
    lr_path      = os.path.join(WORKING_DATA, 'lr_model.pkl')
    best_path    = os.path.join(WORKING_DATA, 'best_model.pkl')
    cluster_path = os.path.join(WORKING_DATA, 'clustering.pkl')

    with open(rf_path,      'rb') as f: rf_data      = pickle.load(f)
    with open(lr_path,      'rb') as f: lr_data      = pickle.load(f)
    with open(best_path,    'rb') as f: best_data    = pickle.load(f)
    with open(cluster_path, 'rb') as f: cluster_data = pickle.load(f)

    metrics = {
        'best_model': best_data['metrics'],
        'random_forest': rf_data['metrics'],
        'logistic_regression': lr_data['metrics'],
        'clustering': {
            'algorithm'       : 'Agglomerative Clustering (Ward linkage)',
            'n_clusters'      : int(cluster_data['best_k']),
            'silhouette_score': round(float(cluster_data['silhouette_score']), 4),
            'cluster_sizes'   : cluster_data['cluster_sizes'],
        },
    }

    metrics_path = os.path.join(WORKING_DATA, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved:\n{json.dumps(metrics, indent=2)}")
    return metrics_path


# ─── Task 2: Generate Evaluation Plots ───────────────────────────────────────

def generate_plots(**context):
    """
    Figure 1 (evaluation_plots.png): Confusion Matrix + ROC Curve
    Figure 2 (feature_importance.png): Top-15 features (Random Forest)
    """
    best_path  = os.path.join(WORKING_DATA, 'best_model.pkl')
    rf_path    = os.path.join(WORKING_DATA, 'rf_model.pkl')
    split_path = os.path.join(WORKING_DATA, 'splits.pkl')

    with open(best_path,  'rb') as f: best_data  = pickle.load(f)
    with open(rf_path,    'rb') as f: rf_data    = pickle.load(f)
    with open(split_path, 'rb') as f: split_data = pickle.load(f)

    y_test     = best_data['y_test']
    y_pred     = best_data['y_pred']
    model      = best_data['model']
    model_name = best_data['metrics']['model_name']
    X_test     = split_data['X_test']
    feat_names = split_data['feature_names']

    # ── Figure 1: Confusion Matrix + ROC ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Model Evaluation — {model_name}', fontsize=14, fontweight='bold')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Safe', 'Hazardous']).plot(
        ax=axes[0], colorbar=False, cmap='Blues'
    )
    axes[0].set_title('Confusion Matrix', fontsize=12)

    # ROC Curve
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC  (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'navy', lw=1, linestyle='--',
                     label='Random classifier')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve', fontsize=12)
        axes[1].legend(loc='lower right')
    else:
        axes[1].text(0.5, 0.5, 'ROC not available\n(model has no predict_proba)',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('ROC Curve', fontsize=12)

    plt.tight_layout()
    eval_path = os.path.join(WORKING_DATA, 'evaluation_plots.png')
    plt.savefig(eval_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plots saved → {eval_path}")

    # ── Figure 2: Feature Importance (RF only) ────────────────────────────────
    rf_model = rf_data['model']
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]   # top 15
        top_features = [feat_names[i] for i in indices]
        top_values   = importances[indices]

        fig2, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_features)), top_values[::-1],
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features))))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title('Top Feature Importances — Random Forest', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        fi_path = os.path.join(WORKING_DATA, 'feature_importance.png')
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved → {fi_path}")

    return eval_path
