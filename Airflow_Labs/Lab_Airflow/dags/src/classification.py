"""
classification.py
-----------------
Trains two classifiers in parallel on the Air Quality dataset:
  • Random Forest Classifier
  • Logistic Regression

A BranchPythonOperator then picks the winner by weighted F1 score
and saves it as the canonical best_model.pkl.
"""

import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

WORKING_DATA = '/opt/airflow/working_data'
MODEL_DIR    = '/opt/airflow/model'


# ─── Helper ──────────────────────────────────────────────────────────────────

def _load_splits(split_path):
    with open(split_path, 'rb') as f:
        return pickle.load(f)


def _compute_metrics(name, y_test, y_pred):
    return {
        'model_name': name,
        'accuracy' : float(accuracy_score(y_test, y_pred)),
        'f1'       : float(f1_score(y_test, y_pred, average='weighted')),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall'   : float(recall_score(y_test, y_pred, average='weighted')),
    }


# ─── Task 1: Random Forest ────────────────────────────────────────────────────

def train_random_forest(**context):
    """Train a Random Forest (100 trees) and persist model + metrics."""
    split_path = context['ti'].xcom_pull(task_ids='preprocessing_group.split_data')
    data = _load_splits(split_path)

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    print("Training Random Forest (n_estimators=100)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = _compute_metrics('Random Forest', y_test, y_pred)

    print(f"RF Metrics : {metrics}")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Hazardous']))

    rf_path = os.path.join(WORKING_DATA, 'rf_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump({'model': rf, 'metrics': metrics,
                     'y_test': y_test, 'y_pred': y_pred}, f)

    return rf_path


# ─── Task 2: Logistic Regression ─────────────────────────────────────────────

def train_logistic_regression(**context):
    """Train a Logistic Regression (max_iter=1000) and persist model + metrics."""
    split_path = context['ti'].xcom_pull(task_ids='preprocessing_group.split_data')
    data = _load_splits(split_path)

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    print("Training Logistic Regression (max_iter=1000)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    metrics = _compute_metrics('Logistic Regression', y_test, y_pred)

    print(f"LR Metrics : {metrics}")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Hazardous']))

    lr_path = os.path.join(WORKING_DATA, 'lr_model.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump({'model': lr, 'metrics': metrics,
                     'y_test': y_test, 'y_pred': y_pred}, f)

    return lr_path


# ─── Task 3: Branch on Best F1 ───────────────────────────────────────────────

def branch_best_model(**context):
    """
    Compare weighted F1 of both models.
    Returns the task_id of the winner so Airflow follows that branch.
    """
    rf_path = context['ti'].xcom_pull(task_ids='classification_group.train_random_forest')
    lr_path = context['ti'].xcom_pull(task_ids='classification_group.train_logistic_regression')

    with open(rf_path, 'rb') as f:
        rf_f1 = pickle.load(f)['metrics']['f1']

    with open(lr_path, 'rb') as f:
        lr_f1 = pickle.load(f)['metrics']['f1']

    print(f"Random Forest F1      : {rf_f1:.4f}")
    print(f"Logistic Regression F1: {lr_f1:.4f}")

    if rf_f1 >= lr_f1:
        print("Winner → Random Forest")
        return 'classification_group.save_random_forest'
    else:
        print("Winner → Logistic Regression")
        return 'classification_group.save_logistic_regression'


# ─── Task 4a: Save Random Forest ─────────────────────────────────────────────

def save_random_forest(**context):
    """Copy RF results to best_model.pkl and model/model.sav."""
    rf_path = context['ti'].xcom_pull(task_ids='classification_group.train_random_forest')

    with open(rf_path, 'rb') as f:
        rf_data = pickle.load(f)

    best_path = os.path.join(WORKING_DATA, 'best_model.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(rf_data, f)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_sav = os.path.join(MODEL_DIR, 'model.sav')
    with open(model_sav, 'wb') as f:
        pickle.dump(rf_data['model'], f)

    print(f"Saved best model (Random Forest, F1={rf_data['metrics']['f1']:.4f})")
    return best_path


# ─── Task 4b: Save Logistic Regression ───────────────────────────────────────

def save_logistic_regression(**context):
    """Copy LR results to best_model.pkl and model/model.sav."""
    lr_path = context['ti'].xcom_pull(task_ids='classification_group.train_logistic_regression')

    with open(lr_path, 'rb') as f:
        lr_data = pickle.load(f)

    best_path = os.path.join(WORKING_DATA, 'best_model.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(lr_data, f)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_sav = os.path.join(MODEL_DIR, 'model.sav')
    with open(model_sav, 'wb') as f:
        pickle.dump(lr_data['model'], f)

    print(f"Saved best model (Logistic Regression, F1={lr_data['metrics']['f1']:.4f})")
    return best_path
