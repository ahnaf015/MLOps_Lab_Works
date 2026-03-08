"""
Heart Disease Risk Predictor — Flask API Server
================================================
Loads trained model artifacts from /exchange (shared Docker volume)
and serves a medical-themed web dashboard on port 80.

Routes:
  GET  /          → Web dashboard (predict.html)
  POST /predict   → JSON inference endpoint
  GET  /health    → Container health check
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─── Load Model Artifacts from Shared Volume ─────────────────────────────────
MODEL_DIR = "/exchange"

print("Loading model artifacts from /exchange ...")
model  = joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_info.json")) as f:
    feature_info = json.load(f)

with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
    feature_importance = json.load(f)

FEATURE_NAMES = feature_info["names"]
print(f"✓ Model loaded  | Accuracy: {feature_info['accuracy']*100:.2f}%  | AUC: {feature_info['auc']:.4f}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the prediction dashboard."""
    return render_template(
        "predict.html",
        model_accuracy=round(feature_info["accuracy"] * 100, 2),
        model_auc=feature_info["auc"],
        feature_importance=feature_importance,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON body with one key per feature name.
    Return prediction, risk percentage, and risk level.
    """
    try:
        data = request.get_json(force=True)

        # Build feature vector in the exact order used during training
        features = [float(data[name]) for name in FEATURE_NAMES]
        features_array  = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        prediction  = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]

        risk_pct     = round(float(probabilities[1]) * 100, 1)
        no_risk_pct  = round(float(probabilities[0]) * 100, 1)

        # Classify into three risk bands
        if risk_pct < 30:
            risk_level = "Low Risk"
            risk_color = "#2ecc71"   # green
        elif risk_pct < 60:
            risk_level = "Moderate Risk"
            risk_color = "#f39c12"   # orange
        else:
            risk_level = "High Risk"
            risk_color = "#e74c3c"   # red

        return jsonify({
            "prediction":   prediction,
            "risk_pct":     risk_pct,
            "no_risk_pct":  no_risk_pct,
            "risk_level":   risk_level,
            "risk_color":   risk_color,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint — useful for container orchestration."""
    return jsonify({
        "status":   "healthy",
        "model":    "GradientBoostingClassifier",
        "accuracy": feature_info["accuracy"],
        "auc":      feature_info["auc"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
