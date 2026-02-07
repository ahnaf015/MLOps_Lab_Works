# /src/train.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from datetime import datetime
from pathlib import Path
import logging
from data import load_data, preprocess_data, split_data
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):

    logger.info("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=settings.N_ESTIMATORS,
        max_depth=settings.MAX_DEPTH,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    logger.info(f"Model Metrics: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
    return metrics

def save_model_artifacts(model, scaler, metrics):

    # Create model directory if it doesn't exist
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = settings.MODEL_DIR / f"{settings.MODEL_NAME}_{settings.MODEL_VERSION}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = settings.MODEL_DIR / f"scaler_{settings.MODEL_VERSION}.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_name': settings.MODEL_NAME,
        'version': settings.MODEL_VERSION,
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
        'hyperparameters': {
            'n_estimators': settings.N_ESTIMATORS,
            'max_depth': settings.MAX_DEPTH,
            'random_state': settings.RANDOM_STATE
        }
    }
    
    metadata_path = settings.MODEL_DIR / f"metadata_{settings.MODEL_VERSION}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {metadata_path}")

def main():

    logger.info("Starting training pipeline...")
    
    # Load and preprocess data
    X, y = load_data()
    X_scaled, y, scaler = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    save_model_artifacts(model, scaler, metrics)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()