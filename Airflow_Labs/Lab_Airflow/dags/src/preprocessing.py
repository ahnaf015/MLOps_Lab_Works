"""
preprocessing.py
----------------
Handles data loading, validation, cleaning, feature engineering,
and train/test splitting for the UCI Air Quality dataset.

Data source: UCI Machine Learning Repository (direct ZIP download)
  https://archive.ics.uci.edu/static/public/360/air+quality.zip
Missing values in this dataset are coded as -200.
Target: binary label derived from CO(GT) — safe (0) vs hazardous (1).
"""

import io
import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

WORKING_DATA = '/opt/airflow/working_data'

# Direct download URL for UCI Air Quality dataset
UCI_URL = 'https://archive.ics.uci.edu/static/public/360/air+quality.zip'


# ─── Task 1: Load Data ────────────────────────────────────────────────────────

def load_data(**context):
    """
    Download UCI Air Quality dataset directly from the UCI archive.
    The ZIP contains AirQualityUCI.csv (semicolon-separated, comma decimals).
    Caches the raw CSV in working_data so re-runs don't need to re-download.
    """
    cache_csv = os.path.join(WORKING_DATA, 'air_quality_raw.csv')

    if os.path.exists(cache_csv):
        print(f"Loading cached dataset from {cache_csv}")
        df = pd.read_csv(cache_csv, sep=';', decimal=',')
    else:
        print(f"Downloading UCI Air Quality dataset from {UCI_URL} ...")
        response = requests.get(UCI_URL, timeout=120, verify=False)
        response.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(response.content))
        csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
        print(f"Files in ZIP: {zf.namelist()}")
        print(f"Reading: {csv_files[0]}")

        df = pd.read_csv(zf.open(csv_files[0]), sep=';', decimal=',')
        df.to_csv(cache_csv, index=False, sep=';', decimal=',')
        print(f"Dataset cached → {cache_csv}")

    # Drop fully empty trailing rows (last 2 rows in this dataset are blank)
    df = df.dropna(how='all')

    print(f"Raw shape : {df.shape}")
    print(f"Columns   : {df.columns.tolist()}")

    raw_path = os.path.join(WORKING_DATA, 'raw.pkl')
    with open(raw_path, 'wb') as f:
        pickle.dump(df, f)

    print(f"Raw data saved → {raw_path}")
    return raw_path


# ─── Task 2: Validate Data ───────────────────────────────────────────────────

def validate_data(**context):
    """Basic sanity checks on the loaded dataset."""
    raw_path = context['ti'].xcom_pull(task_ids='data_group.load_data')

    with open(raw_path, 'rb') as f:
        df = pickle.load(f)

    assert df.shape[0] > 0, "Dataset is empty!"
    assert df.shape[1] > 0, "No columns found!"

    print(f"Shape        : {df.shape}")
    print(f"Dtypes       :\n{df.dtypes}")
    print(f"Null counts  :\n{df.isnull().sum()}")
    print("Validation passed.")

    return raw_path


# ─── Task 3: Clean Data ───────────────────────────────────────────────────────

def clean_data(**context):
    """
    - Replace -200 (missing sentinel) with NaN
    - Keep only numeric columns
    - Drop columns with >50% missing
    - Fill remaining NaN with column median
    """
    raw_path = context['ti'].xcom_pull(task_ids='data_group.validate_data')

    with open(raw_path, 'rb') as f:
        df = pickle.load(f)

    # Replace missing value sentinel
    df = df.replace(-200, np.nan).replace(-200.0, np.nan)

    # Keep numeric only (drops Date, Time columns)
    df = df.select_dtypes(include=[np.number])

    # Drop columns with more than 50% missing
    thresh = len(df) * 0.5
    before = df.shape[1]
    df = df.dropna(thresh=thresh, axis=1)
    print(f"Dropped {before - df.shape[1]} columns with >50% missing values")

    # Fill remaining NaN with column median
    df = df.fillna(df.median())

    # Drop any remaining all-NaN rows
    df = df.dropna()

    print(f"Clean shape  : {df.shape}")
    print(f"Columns kept : {df.columns.tolist()}")

    clean_path = os.path.join(WORKING_DATA, 'clean.pkl')
    with open(clean_path, 'wb') as f:
        pickle.dump(df, f)

    return clean_path


# ─── Task 4: Feature Engineering + Scaling ───────────────────────────────────

def engineer_and_scale(**context):
    """
    - Identify CO(GT) column as target basis (median split → binary label)
    - StandardScale all feature columns
    - Save X (scaled DataFrame), y (binary Series), scaler, feature names
    """
    clean_path = context['ti'].xcom_pull(task_ids='preprocessing_group.clean_data')

    with open(clean_path, 'rb') as f:
        df = pickle.load(f)

    # Find the CO ground-truth column (CO(GT) or similar)
    co_candidates = [c for c in df.columns if 'CO' in c.upper() and 'GT' in c.upper()]
    target_col = co_candidates[0] if co_candidates else df.columns[0]

    print(f"Target column : {target_col}")

    # Binary target: above median CO level = hazardous (1)
    threshold = df[target_col].median()
    y = (df[target_col] > threshold).astype(int)
    X = df.drop(columns=[target_col])

    print(f"Threshold     : {threshold:.4f}")
    print(f"Class balance : {y.value_counts().to_dict()}")
    print(f"Features      : {X.columns.tolist()}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    feature_path = os.path.join(WORKING_DATA, 'features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump({
            'X': X_scaled,
            'y': y,
            'scaler': scaler,
            'feature_names': X.columns.tolist(),
            'target_col': target_col,
            'threshold': threshold,
        }, f)

    return feature_path


# ─── Task 5: Train / Test Split ──────────────────────────────────────────────

def split_data(**context):
    """80/20 stratified train-test split."""
    feature_path = context['ti'].xcom_pull(
        task_ids='preprocessing_group.engineer_and_scale'
    )

    with open(feature_path, 'rb') as f:
        data = pickle.load(f)

    X, y = data['X'], data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size : {X_train.shape}")
    print(f"Test size  : {X_test.shape}")

    split_path = os.path.join(WORKING_DATA, 'splits.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': data['feature_names'],
            'scaler': data['scaler'],
        }, f)

    return split_path
