"""
dashboard_dag.py
----------------
DAG: Air_Quality_Dashboard

Triggered by Air_Quality_Pipeline (TriggerDagRunOperator) after all tasks
complete.  Starts a Flask web server on port 5555 inside the Celery worker.

Routes:
  GET /           → redirect to /dashboard or /failure
  GET /dashboard  → full metrics + embedded plots + DAG metadata
  GET /success    → simple success page
  GET /failure    → failure page with run details
  GET /metrics    → JSON API — model + clustering metrics
  GET /plot/confusion     → evaluation_plots.png (confusion matrix + ROC)
  GET /plot/dendrogram    → dendrogram.png (cluster plots)
  GET /plot/importance    → feature_importance.png (RF importance)
  GET /health     → {"status": "ok"}

The dashboard auto-refreshes every 30 seconds and polls the Airflow REST API
to get the latest run state of Air_Quality_Pipeline.
"""

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


def start_flask_dashboard(**context):
    import os
    import json
    from flask import Flask, render_template, jsonify, send_file, redirect, url_for
    import requests
    from requests.auth import HTTPBasicAuth

    WORKING_DATA = '/opt/airflow/working_data'
    AIRFLOW_API  = 'http://airflow-webserver:8080/api/v1'
    AUTH         = HTTPBasicAuth('airflow2', 'airflow2')
    PIPELINE_DAG = 'Air_Quality_Pipeline'

    app = Flask(
        __name__,
        template_folder='/opt/airflow/dags/templates',
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_latest_run():
        """Return the most recent DagRun dict for Air_Quality_Pipeline, or None."""
        try:
            resp = requests.get(
                f'{AIRFLOW_API}/dags/{PIPELINE_DAG}/dagRuns',
                auth=AUTH,
                params={'order_by': '-execution_date', 'limit': 1},
                timeout=5,
            )
            runs = resp.json().get('dag_runs', [])
            return runs[0] if runs else None
        except Exception as exc:
            print(f"[Dashboard] Airflow API error: {exc}")
            return None

    def load_metrics():
        """Load metrics.json if it exists."""
        path = os.path.join(WORKING_DATA, 'metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def plot_exists(filename):
        return os.path.exists(os.path.join(WORKING_DATA, filename))

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route('/')
    def index():
        run = get_latest_run()
        if run is None:
            return render_template('failure.html', run=None,
                                   error='No DAG run found. Run the pipeline first.')
        state = run.get('state', 'unknown')
        if state == 'success':
            return redirect('/dashboard')
        elif state == 'failed':
            return render_template('failure.html', run=run, error=None)
        else:
            return render_template('success.html', run=run, status='in_progress')

    @app.route('/dashboard')
    def dashboard():
        run     = get_latest_run()
        metrics = load_metrics()
        plots   = {
            'confusion'  : plot_exists('evaluation_plots.png'),
            'dendrogram' : plot_exists('dendrogram.png'),
            'importance' : plot_exists('feature_importance.png'),
        }
        return render_template('dashboard.html',
                               run=run, metrics=metrics, plots=plots)

    @app.route('/success')
    def success():
        run = get_latest_run()
        return render_template('success.html', run=run, status='success')

    @app.route('/failure')
    def failure():
        run = get_latest_run()
        return render_template('failure.html', run=run, error=None)

    @app.route('/metrics')
    def metrics_api():
        metrics = load_metrics()
        if metrics:
            return jsonify(metrics)
        return jsonify({'error': 'Metrics not yet available. Run the pipeline first.'}), 404

    @app.route('/plot/confusion')
    def plot_confusion():
        path = os.path.join(WORKING_DATA, 'evaluation_plots.png')
        if os.path.exists(path):
            return send_file(path, mimetype='image/png')
        return 'Plot not generated yet. Run the pipeline first.', 404

    @app.route('/plot/dendrogram')
    def plot_dendrogram():
        path = os.path.join(WORKING_DATA, 'dendrogram.png')
        if os.path.exists(path):
            return send_file(path, mimetype='image/png')
        return 'Plot not generated yet. Run the pipeline first.', 404

    @app.route('/plot/importance')
    def plot_importance():
        path = os.path.join(WORKING_DATA, 'feature_importance.png')
        if os.path.exists(path):
            return send_file(path, mimetype='image/png')
        return 'Feature importance plot not available.', 404

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'service': 'Air Quality Dashboard'})

    # ── Start ─────────────────────────────────────────────────────────────────
    print("[Dashboard] Flask server starting on 0.0.0.0:5555 ...")
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id='Air_Quality_Dashboard',
    description='Flask dashboard — triggered by Air_Quality_Pipeline',
    schedule_interval=None,
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    tags=['dashboard', 'flask', 'air-quality', 'lab'],
) as dag:

    start_dashboard = PythonOperator(
        task_id='start_flask_dashboard',
        python_callable=start_flask_dashboard,
    )
