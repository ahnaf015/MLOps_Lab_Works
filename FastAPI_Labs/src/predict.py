# /src/predict.py

import joblib
import numpy as np
import logging
from pathlib import Path
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_artifacts()
    
    def load_artifacts(self):
        
        try:
            model_path = settings.MODEL_DIR / f"{settings.MODEL_NAME}_{settings.MODEL_VERSION}.pkl"
            scaler_path = settings.MODEL_DIR / f"scaler_{settings.MODEL_VERSION}.pkl"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Model and scaler loaded successfully from {settings.MODEL_DIR}")
        
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> float:

        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        
        return prediction[0]

# Global predictor instance
predictor = WineQualityPredictor()

def predict_quality(features: list) -> float:

    features_array = np.array(features).reshape(1, -1)
    return predictor.predict(features_array)