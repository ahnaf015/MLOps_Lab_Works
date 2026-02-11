# LAB 1 - GitHub (Lab2): Automated Model Training & Calibration with GitHub Actions and MLflow

## Overview

This lab demonstrates an end-to-end MLOps workflow using **GitHub Actions** for CI/CD automation and **MLflow** for experiment tracking. The project trains and evaluates machine learning models on two datasets — **Breast Cancer Wisconsin** (automated pipeline) and **Wine Quality** (notebook-based experimentation) — with automated retraining triggered on push events and scheduled daily calibration.

---

## Project Structure

```
Github_Labs/Lab2/
├── metrics/                  # Stored evaluation metrics (JSON)
├── mlruns/                   # MLflow experiment tracking artifacts
├── models/                   # Saved trained model files (.joblib)
├── src/
│   ├── data/                 # Cached dataset pickle files
│   ├── metrics/              # Source-level metrics
│   ├── mlruns/               # Source-level MLflow runs
│   ├── __init__.py
│   ├── evaluate_model.py     # Model evaluation script
│   ├── test.ipynb            # Wine Quality experiment notebook
│   └── train_model.py        # Breast Cancer training pipeline
├── test/
│   └── __init__.py
├── workflows/
│   ├── model_calibration.yml            # Scheduled daily retraining
│   └── model_calibration_on_push.yml    # Retraining on push to main
├── README.md
└── requirements.txt
```

---

## Datasets & Models

### 1. Breast Cancer Wisconsin (Automated Pipeline)
- **Script:** `src/train_model.py`
- **Model:** Logistic Regression with StandardScaler (Pipeline)
- **Split:** 70% Train / 15% Validation / 15% Test
- **Threshold Tuning:** F1-optimized threshold search on the validation set
- **Metrics Logged:** Validation F1, Test Accuracy, Test F1

### 2. Wine Quality Classification (Notebook)
- **Notebook:** `src/test.ipynb`
- **Model:** Gradient Boosting Classifier
- **Split:** 70% Train / 30% Test
- **Performance:** ~96.3% Accuracy, ~96.5% F1 (Macro)
- **Metrics Logged:** Accuracy, F1 Score (Macro & Weighted)

---

## GitHub Actions Workflows

### Retraining on Push (`model_calibration_on_push.yml`)
Triggers automatically on every push to the `main` branch. Steps:
1. Checkout repository
2. Set up Python 3.9 and install dependencies
3. Generate a unique timestamp
4. Retrain the model using `train_model.py`
5. Evaluate the model using `evaluate_model.py`
6. Verify model and metrics artifacts exist
7. Commit and push updated model + metrics back to `main`

### Scheduled Daily Calibration (`model_calibration.yml`)
Runs on a daily cron schedule (`0 0 * * *`) and can also be triggered manually via `workflow_dispatch`. Follows the same pipeline as above to ensure the model stays calibrated.

---

## MLflow Experiment Tracking

Both scripts log experiments to MLflow with the following tracked information:
- **Parameters:** Dataset name, sample count, feature count, model type, split sizes, hyperparameters
- **Metrics:** Accuracy, F1 Score, threshold-tuned metrics
- **Artifacts:** Trained model files (`.joblib`)

### Viewing MLflow UI
```bash
# From the src/ directory:
python -m mlflow ui --backend-store-uri ./mlruns

# Or from Lab2/ directory:
python -m mlflow ui --backend-store-uri ./src/mlruns
```
Then navigate to **http://localhost:5000**

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ahnaf015/MLOps_lab_works.git
cd MLOps_lab_works/Github_Labs/Lab2
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- scikit-learn
- numpy
- pandas
- pytest
- ipykernel
- mlflow

---

## Usage

### Run Training Manually
```bash
python src/train_model.py --timestamp "$(date '+%Y%m%d%H%M%S')"
```

### Run the Wine Quality Notebook
```bash
jupyter notebook src/test.ipynb
```

---

## Key Features

- **Automated CI/CD Pipeline** — Model retraining triggered on every push to `main`
- **Scheduled Retraining** — Daily cron job for periodic model calibration
- **Experiment Tracking** — Full MLflow integration for parameters, metrics, and artifacts
- **Threshold Optimization** — F1-score-based threshold tuning on validation data
- **Artifact Versioning** — Timestamped model files committed back to the repository

---

## License

This project is licensed under the terms of the [MIT License](../../LICENSE).
