# ============================================================
# Multi-Stage Dockerfile — Heart Disease Risk Predictor
# ============================================================
# Stage 1 trains the GradientBoosting model and saves it to a
# shared Docker volume (/exchange). Stage 2 picks up that model
# and runs the Flask prediction API on port 80.
# ============================================================


# ─────────────────────────────────────────────────────────────
# Stage 1 : Model Training
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS trainer

LABEL stage="trainer"
LABEL description="Downloads UCI heart disease data, trains GradientBoostingClassifier, saves artifacts to /exchange"

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the training script
COPY src/model_training.py .

# Run training — artifacts are written to /exchange (mounted volume)
CMD ["python", "model_training.py"]


# ─────────────────────────────────────────────────────────────
# Stage 2 : Flask API Server
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS server

LABEL stage="server"
LABEL description="Serves the trained model via a Flask REST API with a medical-themed dashboard"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/main.py .
COPY src/templates/ templates/

EXPOSE 80

CMD ["python", "main.py"]
