# Lab_Airflow — Air Quality Analysis Pipeline

An end-to-end MLOps pipeline built with **Apache Airflow**, combining unsupervised and supervised machine learning on the **UCI Air Quality dataset**, served through a live **Flask dashboard**.

This lab is a combination of concepts from previous labs — where Lab 1 focused on getting Airflow running with a basic ML pipeline, and Lab 2 introduced email notifications and a Flask web interface. This lab takes those foundations further by introducing advanced Airflow features, a richer ML pipeline with automatic model selection, and a full interactive dashboard.

---

## What This Lab Does

### Machine Learning Pipeline
- **Unsupervised:** Agglomerative (Hierarchical) Clustering with Ward linkage to discover natural pollution patterns in sensor data
- **Supervised:** Trains two classifiers (Random Forest and Logistic Regression) in parallel and automatically selects the best one based on weighted F1 score
- **Evaluation:** Generates confusion matrix, ROC curve, and feature importance plots

### Advanced Airflow Features
- **TaskGroup** — related tasks are logically grouped (data, preprocessing, clustering, classification, reporting)
- **BranchPythonOperator** — dynamically routes execution to the winning model based on F1 score
- **TriggerRule** — correctly handles skipped branch tasks using `NONE_FAILED_MIN_ONE_SUCCESS`
- **SLA** — per-task deadline with a miss callback
- **TriggerDagRunOperator** — fires the dashboard DAG automatically when the pipeline finishes
- **EmailOperator** — sends failure/retry notifications
- **owner_links** — clickable owner in the Airflow UI

### Flask Dashboard
- Live metrics display (accuracy, F1, precision, recall)
- Side-by-side model comparison table
- Clustering stats (optimal k, silhouette score, cluster sizes)
- Three embedded plots: confusion matrix + ROC, dendrogram + PCA scatter, feature importances
- Auto-refreshes every 30 seconds
- JSON metrics API at `/metrics`

---

## Architecture

```
[startup_banner]  (BashOperator)
        ↓
[data_group]
  load_data → validate_data
        ↓
[preprocessing_group]
  clean_data → engineer_and_scale → split_data
        ↓                 ↓
[clustering_group]   [classification_group]
  train_agglomerative    train_random_forest ─┐
  → generate_cluster_plots                    ├→ branch_best_model
                         train_logistic_reg ──┘      ↓          ↓
                                              save_rf      save_lr
        ↓                         ↓
[reporting_group]
  generate_metrics → generate_plots
        ↓
[trigger_dashboard]  →  Air_Quality_Dashboard DAG
                              ↓
                      Flask server (port 5555)
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with at least **4 GB RAM** allocated)
- [Docker Compose](https://docs.docker.com/compose/)
- Internet connection (to download the dataset on first run via OpenML)

---

## Setup & Launch

### 1. Navigate to the lab directory

```bash
cd Airflow_Labs/Lab_Airflow
```

### 2. Run the setup script

```bash
bash setup.sh
```

This will:
- Create required directories (`logs/`, `working_data/`, `model/`)
- Run `docker-compose up airflow-init` (DB migration + admin user creation)
- Start all services in detached mode

> **Windows users:** Run `setup.sh` inside Git Bash or WSL.

### 3. Wait for services to be healthy

```bash
docker-compose ps
```

All services should show `healthy` before proceeding (~2–3 minutes on first run).

---

## Running the Pipeline

### Step 1 — Open Airflow UI

Navigate to [http://localhost:8082](http://localhost:8082)

```
Username: airflow2
Password: airflow2
```

### Step 2 — Unpause the DAG

Find **`Air_Quality_Pipeline`** in the DAG list and toggle it **ON** (unpause).

### Step 3 — Trigger manually

Click the **play button (▶)** → "Trigger DAG" to run it now.

### Step 4 — Watch progress

Go to **Graph** view to see tasks completing in real time. The pipeline takes ~3–5 minutes on first run (dataset download included).

### Step 5 — Open the Dashboard

Once the pipeline finishes, visit [http://localhost:5050](http://localhost:5050)

---

## Dashboard Routes

| Route | Description |
|---|---|
| `/` | Redirects based on latest run state |
| `/dashboard` | Full metrics + plots (auto-refreshes every 30s) |
| `/success` | Simple success page |
| `/failure` | Failure page with run details |
| `/metrics` | Raw JSON — all model and clustering metrics |
| `/plot/confusion` | Confusion matrix + ROC curve (PNG) |
| `/plot/dendrogram` | Dendrogram + PCA cluster scatter (PNG) |
| `/plot/importance` | Top feature importances — Random Forest (PNG) |
| `/health` | Health check `{"status": "ok"}` |

---

## Pipeline Tasks Explained

### data_group
| Task | What it does |
|---|---|
| `load_data` | Downloads UCI Air Quality via `sklearn.fetch_openml`, saves raw pickle |
| `validate_data` | Checks shape, dtypes, null counts — raises assertion if empty |

### preprocessing_group
| Task | What it does |
|---|---|
| `clean_data` | Replaces -200 (missing sentinel), drops high-null columns, median-fills rest |
| `engineer_and_scale` | Builds binary target from CO(GT) median split, StandardScales all features |
| `split_data` | 80/20 stratified train/test split |

### clustering_group
| Task | What it does |
|---|---|
| `train_agglomerative` | Sweeps k=2..6, picks best k by silhouette score, fits Ward linkage model |
| `generate_cluster_plots` | Saves dendrogram + PCA scatter to `working_data/dendrogram.png` |

### classification_group
| Task | What it does |
|---|---|
| `train_random_forest` | Trains RF (100 trees), saves model + metrics |
| `train_logistic_regression` | Trains LR (max_iter=1000), saves model + metrics |
| `branch_best_model` | **BranchPythonOperator** — compares weighted F1, routes to winner |
| `save_random_forest` | (if RF wins) copies to `best_model.pkl` + `model/model.sav` |
| `save_logistic_regression` | (if LR wins) copies to `best_model.pkl` + `model/model.sav` |

### reporting_group
| Task | What it does |
|---|---|
| `generate_metrics` | Writes `working_data/metrics.json` with all scores |
| `generate_plots` | Saves confusion matrix + ROC curve + feature importance PNGs |

### trigger_dashboard
Fires `Air_Quality_Dashboard` DAG (non-blocking), which starts the Flask server on port 5555.

---

## Dataset

**UCI Air Quality** — Hourly averaged responses from metal-oxide chemical sensors deployed in a heavily polluted area.

- **Source:** `sklearn.datasets.fetch_openml('air-quality', version=1)` — auto-downloads, no manual steps
- **Samples:** ~9,358 rows
- **Features:** CO, NOx, NO2, Benzene (C6H6), Temperature, Humidity, etc.
- **Missing values:** coded as `-200` — cleaned automatically in the pipeline
- **Target:** Binary — CO(GT) above median = hazardous (1), below = safe (0)

---

## Output Files

All outputs land in `working_data/` (mounted as a Docker volume):

| File | Description |
|---|---|
| `raw.pkl` | Raw dataset from OpenML |
| `clean.pkl` | After -200 removal and null handling |
| `features.pkl` | Scaled X, binary y, scaler, feature names |
| `splits.pkl` | X_train, X_test, y_train, y_test |
| `clustering.pkl` | Fitted AgglomerativeClustering + labels + scores |
| `rf_model.pkl` | Random Forest model + metrics |
| `lr_model.pkl` | Logistic Regression model + metrics |
| `best_model.pkl` | Winner model (copied from rf or lr) |
| `metrics.json` | All metrics in JSON — served at `/metrics` |
| `dendrogram.png` | Dendrogram + PCA cluster scatter |
| `evaluation_plots.png` | Confusion matrix + ROC curve |
| `feature_importance.png` | Top-15 RF feature importances |

The best model is also saved to `model/model.sav`.

---

## Stopping the Services

```bash
docker-compose down
```

To also remove the database volume:

```bash
docker-compose down -v
```

---

## Screenshots

Screenshots are located in the `screenshots/` folder.

| File | Description |
|---|---|
| `01_dag_graph.png` | DAG graph view with TaskGroups |
| `02_dag_running.png` | Tasks in progress |
| `03_dag_success.png` | All tasks green |
| `04_dashboard_main.png` | Flask dashboard full view |
| `05_confusion_matrix.png` | Confusion matrix + ROC curve |
| `06_dendrogram.png` | Clustering dendrogram + PCA scatter |
| `07_feature_importance.png` | Feature importance bar chart |
| `08_metrics_json.png` | Raw /metrics JSON endpoint |

---

## Author

**Mohammed Ahnaf Tajwar**
