"""
Main orchestrator for the Mushroom Safety Classifier pipeline.
Runs all phases sequentially, then launches the API and UI.

Usage:
    python main.py                    # Run all phases + launch servers
    python main.py --phase 1          # Run only Phase 1 (EDA)
    python main.py --phase 2          # Run only Phase 2 (Training)
    python main.py --phase 2 --sweep  # Run Phase 2 with W&B Sweep
    python main.py --phase 3          # Run only Phase 3 (Registry)
    python main.py --serve            # Launch API + UI only (skip training)
"""

import os
import sys
import argparse
import subprocess
import time

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(__file__))


def run_phase1():
    """Phase 1: EDA + W&B Artifacts & Tables."""
    print("=" * 60)
    print("  PHASE 1: Exploratory Data Analysis")
    print("=" * 60)
    from phase1_eda import run_eda
    run_eda()
    print()


def run_phase2(sweep=False, sweep_count=10, model_type="xgboost"):
    """Phase 2: Model Training + W&B Sweeps."""
    print("=" * 60)
    print("  PHASE 2: Model Training")
    print("=" * 60)
    from phase2_train import run_sweep, train_single
    if sweep:
        run_sweep(count=sweep_count)
    else:
        train_single(model_type=model_type)
    print()


def run_phase3():
    """Phase 3: Register best model in W&B Model Registry."""
    print("=" * 60)
    print("  PHASE 3: Model Registry")
    print("=" * 60)
    from phase3_registry import register_model
    register_model()
    print()


def launch_servers():
    """Phase 4 & 5: Launch FastAPI + Streamlit."""
    print("=" * 60)
    print("  PHASE 4 & 5: Launching Servers")
    print("=" * 60)

    lab3_dir = os.path.dirname(os.path.abspath(__file__))

    # Start FastAPI in background
    print("[Server] Starting FastAPI on http://localhost:8000 ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=lab3_dir,
    )
    time.sleep(3)

    # Start Gradio in background
    print("[Server] Starting Gradio on http://localhost:7860 ...")
    ui_proc = subprocess.Popen(
        [sys.executable, "app/ui.py"],
        cwd=lab3_dir,
    )

    print()
    print("=" * 60)
    print("  SERVERS RUNNING")
    print("  FastAPI:   http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("  Gradio:    http://localhost:7860")
    print("  Press Ctrl+C to stop all servers")
    print("=" * 60)

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        api_proc.terminate()
        ui_proc.terminate()
        api_proc.wait()
        ui_proc.wait()
        print("[Server] All servers stopped.")


def main():
    parser = argparse.ArgumentParser(description="Mushroom Safety Classifier Pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run a specific phase only")
    parser.add_argument("--sweep", action="store_true", help="Use W&B Sweep in Phase 2")
    parser.add_argument("--sweep-count", type=int, default=10, help="Number of sweep runs (default: 10)")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "random_forest"],
                        help="Model type for single training run")
    parser.add_argument("--serve", action="store_true", help="Only launch API + UI servers")
    args = parser.parse_args()

    # Check dataset exists
    from app.utils import DATA_PATH
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please download the mushroom dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/uciml/mushroom-classification")
        print(f"And place 'mushrooms.csv' in the data/ folder.")
        sys.exit(1)

    if args.serve:
        launch_servers()
        return

    if args.phase:
        if args.phase == 1:
            run_phase1()
        elif args.phase == 2:
            run_phase2(sweep=args.sweep, sweep_count=args.sweep_count, model_type=args.model)
        elif args.phase == 3:
            run_phase3()
    else:
        # Run all phases
        run_phase1()
        run_phase2(sweep=args.sweep, sweep_count=args.sweep_count, model_type=args.model)
        run_phase3()
        launch_servers()


if __name__ == "__main__":
    main()
