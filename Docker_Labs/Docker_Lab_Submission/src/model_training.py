"""
Heart Disease Risk Predictor — Model Training
==============================================
Dataset : Cleveland Heart Disease (UCI ML Repository, 303 samples, 13 features)
Model   : Gradient Boosting Classifier (scikit-learn)
Output  : Trained model + scaler + metadata → /exchange (shared Docker volume)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Heart Disease Risk Predictor — Training           ║
║        Dataset  : UCI Cleveland Heart Disease            ║
║        Model    : Gradient Boosting Classifier           ║
╚══════════════════════════════════════════════════════════╝
"""
print(BANNER)

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("[1/5] Downloading Cleveland Heart Disease dataset from UCI ML Repository...")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

try:
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS)
    print(f"    ✓ Downloaded {len(df)} records from UCI repository.")
except Exception as e:
    print(f"    ✗ Download failed ({e}). Exiting.")
    raise SystemExit(1)

# ─── 2. Clean Data ────────────────────────────────────────────────────────────
print("\n[2/5] Cleaning dataset...")

# Replace '?' (missing marker in UCI file) with NaN, then drop those rows
df.replace("?", np.nan, inplace=True)
missing_before = df.isnull().sum().sum()
df.dropna(inplace=True)
df = df.astype(float)

print(f"    Removed {missing_before} missing values → {len(df)} clean samples remain.")

# Binarise target: 0 = no disease, 1-4 = some degree of disease → 1
df["target"] = (df["target"] > 0).astype(int)

pos = int(df["target"].sum())
neg = len(df) - pos
print(f"    Heart Disease Positive : {pos}  ({pos/len(df)*100:.1f}%)")
print(f"    Heart Disease Negative : {neg}  ({neg/len(df)*100:.1f}%)")

# ─── 3. Prepare Features ──────────────────────────────────────────────────────
print("\n[3/5] Preparing features and splitting data...")

FEATURE_NAMES = COLUMNS[:-1]
X = df[FEATURE_NAMES].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"    Train set : {len(X_train)} samples")
print(f"    Test  set : {len(X_test)}  samples")

# ─── 4. Train Model ───────────────────────────────────────────────────────────
print("\n[4/5] Training Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=5,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train_scaled, y_train)
print("    ✓ Training complete.")

# Evaluate
y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n    Accuracy  : {accuracy*100:.2f}%")
print(f"    ROC-AUC   : {auc:.4f}")
print(f"    Confusion Matrix:\n{cm}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

# ─── 5. Save Artifacts ────────────────────────────────────────────────────────
print("[5/5] Saving model artifacts to /exchange ...")

os.makedirs("/exchange", exist_ok=True)

joblib.dump(model,  "/exchange/heart_model.pkl")
joblib.dump(scaler, "/exchange/scaler.pkl")

# Metadata used by the Flask app to build the web form
feature_info = {
    "names":    FEATURE_NAMES,
    "min":      df[FEATURE_NAMES].min().round(2).tolist(),
    "max":      df[FEATURE_NAMES].max().round(2).tolist(),
    "mean":     df[FEATURE_NAMES].mean().round(2).tolist(),
    "accuracy": round(float(accuracy), 4),
    "auc":      round(float(auc), 4),
}

with open("/exchange/feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)

importance_dict = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
with open("/exchange/feature_importance.json", "w") as f:
    json.dump(importance_dict, f, indent=2)

print("    ✓ heart_model.pkl")
print("    ✓ scaler.pkl")
print("    ✓ feature_info.json")
print("    ✓ feature_importance.json")
print("\n✅  All artifacts saved. Serving container can now start.")
