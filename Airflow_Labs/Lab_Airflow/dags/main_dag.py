"""
main_dag.py
-----------
DAG: Air_Quality_Pipeline

End-to-end MLOps pipeline on the UCI Air Quality dataset.

Advanced Airflow features demonstrated:
  • TaskGroup          — logical grouping of related tasks
  • BranchPythonOperator — dynamic branching based on model F1 score
  • TriggerRule         — handle skipped branch tasks correctly
  • SLA                 — per-task deadline with miss callback
  • TriggerDagRunOperator — fire-and-forget launch of dashboard DAG
  • EmailOperator       — failure/completion notifications
  • owner_links         — clickable owner in Airflow UI
  • tags                — filterable labels

Pipeline DAG shape:
  [data_group]
    load_data → validate_data
        ↓
  [preprocessing_group]
    clean_data → engineer_and_scale → split_data
        ↓                    ↓
  [clustering_group]   [classification_group]
    train_agglomerative    train_rf ─┐
    → generate_cluster_plots         ├→ branch_best_model → save_rf
                           train_lr ─┘                   → save_lr
        ↓                    ↓
  [reporting_group]
    generate_metrics → generate_plots
        ↓
  trigger_dashboard (TriggerDagRunOperator)
"""

import sys
sys.path.insert(0, '/opt/airflow/dags')

from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from src.preprocessing import (
    load_data, validate_data, clean_data, engineer_and_scale, split_data,
)
from src.clustering import train_agglomerative, generate_cluster_plots
from src.classification import (
    train_random_forest, train_logistic_regression,
    branch_best_model, save_random_forest, save_logistic_regression,
)
from src.evaluation import generate_metrics, generate_plots


# ─── SLA miss callback ───────────────────────────────────────────────────────

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    print(f"[SLA MISS] DAG={dag.dag_id}  tasks={task_list}")


# ─── DAG defaults ────────────────────────────────────────────────────────────

default_args = {
    'owner'           : 'Mohammed Ahnaf Tajwar',
    'depends_on_past' : False,
    'email'           : ['tajwar.mohammedahnaf@gmail.com'],
    'email_on_failure': True,
    'email_on_retry'  : False,
    'retries'         : 1,
    'retry_delay'     : timedelta(minutes=5),
    'sla'             : timedelta(minutes=30),
}

# ─── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id='Air_Quality_Pipeline',
    default_args=default_args,
    description=(
        'Air Quality Analysis: Agglomerative Clustering + '
        'Random Forest vs Logistic Regression'
    ),
    schedule_interval='@daily',
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    max_active_runs=1,
    sla_miss_callback=sla_miss_callback,
    owner_links={
        'Mohammed Ahnaf Tajwar': 'https://github.com/ahnaf015/MLOps_Lab_Works'
    },
    tags=['ml', 'air-quality', 'clustering', 'classification', 'lab'],
) as dag:

    # ── Startup banner ────────────────────────────────────────────────────────
    startup_banner = BashOperator(
        task_id='startup_banner',
        bash_command=(
            'echo "======================================" && '
            'echo "  Air Quality Pipeline — Starting   " && '
            'echo "  Owner : Mohammed Ahnaf Tajwar      " && '
            'echo "  Date  : $(date)                   " && '
            'echo "======================================"'
        ),
    )

    # ── Group 1: Data Ingestion ───────────────────────────────────────────────
    with TaskGroup('data_group', tooltip='Load and validate raw data') as data_group:

        t_load = PythonOperator(
            task_id='load_data',
            python_callable=load_data,
        )

        t_validate = PythonOperator(
            task_id='validate_data',
            python_callable=validate_data,
        )

        t_load >> t_validate

    # ── Group 2: Preprocessing ────────────────────────────────────────────────
    with TaskGroup('preprocessing_group', tooltip='Clean, scale, split') as preprocess_group:

        t_clean = PythonOperator(
            task_id='clean_data',
            python_callable=clean_data,
        )

        t_engineer = PythonOperator(
            task_id='engineer_and_scale',
            python_callable=engineer_and_scale,
        )

        t_split = PythonOperator(
            task_id='split_data',
            python_callable=split_data,
        )

        t_clean >> t_engineer >> t_split

    # ── Group 3: Clustering ───────────────────────────────────────────────────
    with TaskGroup('clustering_group', tooltip='Agglomerative clustering + plots') as cluster_group:

        t_cluster = PythonOperator(
            task_id='train_agglomerative',
            python_callable=train_agglomerative,
        )

        t_dendro = PythonOperator(
            task_id='generate_cluster_plots',
            python_callable=generate_cluster_plots,
        )

        t_cluster >> t_dendro

    # ── Group 4: Classification ───────────────────────────────────────────────
    with TaskGroup('classification_group', tooltip='RF vs LR, branch on best F1') as classify_group:

        t_rf = PythonOperator(
            task_id='train_random_forest',
            python_callable=train_random_forest,
        )

        t_lr = PythonOperator(
            task_id='train_logistic_regression',
            python_callable=train_logistic_regression,
        )

        t_branch = BranchPythonOperator(
            task_id='branch_best_model',
            python_callable=branch_best_model,
        )

        t_save_rf = PythonOperator(
            task_id='save_random_forest',
            python_callable=save_random_forest,
        )

        t_save_lr = PythonOperator(
            task_id='save_logistic_regression',
            python_callable=save_logistic_regression,
        )

        [t_rf, t_lr] >> t_branch >> [t_save_rf, t_save_lr]

    # ── Group 5: Reporting ────────────────────────────────────────────────────
    with TaskGroup('reporting_group', tooltip='Metrics JSON + evaluation plots') as report_group:

        t_metrics = PythonOperator(
            task_id='generate_metrics',
            python_callable=generate_metrics,
            # One save task will be skipped — this rule handles that correctly
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        t_plots = PythonOperator(
            task_id='generate_plots',
            python_callable=generate_plots,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        t_metrics >> t_plots

    # ── Trigger Dashboard DAG ─────────────────────────────────────────────────
    trigger_dashboard = TriggerDagRunOperator(
        task_id='trigger_dashboard',
        trigger_dag_id='Air_Quality_Dashboard',
        wait_for_completion=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── Pipeline wiring ───────────────────────────────────────────────────────
    startup_banner >> data_group >> preprocess_group
    preprocess_group >> [cluster_group, classify_group]
    [cluster_group, classify_group] >> report_group
    report_group >> trigger_dashboard
