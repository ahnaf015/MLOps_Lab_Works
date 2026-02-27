#!/bin/bash
# =============================================================================
# Lab_Airflow Setup Script
# Air Quality Analysis Pipeline — Agglomerative Clustering + Classification
# =============================================================================

echo "=============================================="
echo "  Lab_Airflow — Air Quality Pipeline Setup"
echo "=============================================="
echo ""

# Create required local directories
echo "[1/3] Creating required directories..."
mkdir -p logs working_data model plugins
echo "      Done."
echo ""

# Initialize Airflow DB and create admin user
echo "[2/3] Initializing Airflow (DB migration + admin user)..."
docker-compose up airflow-init
echo "      Done."
echo ""

# Start all services in detached mode
echo "[3/3] Starting all Airflow services..."
docker-compose up -d


# Runs as root inside the worker container so it can chmod Docker-owned files
echo "      Fixing log directory permissions..."
docker-compose exec -u root -T airflow-worker chmod -R 777 /opt/airflow/logs 2>/dev/null || true
echo "      Done."
echo ""

echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Airflow UI  -->  http://localhost:8082"
echo "  Username    -->  airflow2"
echo "  Password    -->  airflow2"
echo ""
echo "  Dashboard   -->  http://localhost:5050"
echo "  (available after triggering Air_Quality_Pipeline DAG)"
echo ""
echo "  Steps to run the lab:"
echo "    1. Open http://localhost:8082"
echo "    2. Unpause 'Air_Quality_Pipeline' DAG"
echo "    3. Trigger it manually (play button)"
echo "    4. Watch tasks complete in Graph view"
echo "    5. Open http://localhost:5050 for the dashboard"
echo "=============================================="
