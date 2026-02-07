# /src/main.py

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
import logging
from datetime import datetime
from predict import predict_quality
from config import settings
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Predict wine quality based on physicochemical properties"
)

class WineFeatures(BaseModel):

    fixed_acidity: float = Field(..., ge=0, le=20, description="Fixed acidity (g/dm³)")
    volatile_acidity: float = Field(..., ge=0, le=2, description="Volatile acidity (g/dm³)")
    citric_acid: float = Field(..., ge=0, le=1.5, description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., ge=0, le=20, description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., ge=0, le=1, description="Chlorides (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, description="Free SO₂ (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Total SO₂ (mg/dm³)")
    density: float = Field(..., ge=0.98, le=1.01, description="Density (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0, le=2, description="Sulphates (g/dm³)")
    alcohol: float = Field(..., ge=8, le=15, description="Alcohol content (%)")
    
    @validator('*', pre=True)
    def check_not_none(cls, v):
        if v is None:
            raise ValueError('Field cannot be None')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    quality_score: float = Field(..., description="Predicted wine quality (0-10 scale)")
    quality_category: str = Field(..., description="Quality category (Poor/Average/Good/Excellent)")
    confidence: str = Field(..., description="Prediction confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_version: str
    timestamp: str

@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():

    return HealthResponse(
        status="healthy",
        model_version=settings.MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", status_code=status.HTTP_200_OK)
async def model_info():

    try:
        metadata_path = settings.MODEL_DIR / f"metadata_{settings.MODEL_VERSION}.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found"
        )
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model information"
        )

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_wine_quality(wine_features: WineFeatures):

    try:
        # Extract features in correct order
        features = [
            wine_features.fixed_acidity,
            wine_features.volatile_acidity,
            wine_features.citric_acid,
            wine_features.residual_sugar,
            wine_features.chlorides,
            wine_features.free_sulfur_dioxide,
            wine_features.total_sulfur_dioxide,
            wine_features.density,
            wine_features.pH,
            wine_features.sulphates,
            wine_features.alcohol
        ]
        
        # Make prediction
        quality_score = predict_quality(features)
        
        # Categorize quality
        if quality_score < 5:
            category = "Poor"
            confidence = "Medium"
        elif quality_score < 6:
            category = "Average"
            confidence = "High"
        elif quality_score < 7:
            category = "Good"
            confidence = "High"
        else:
            category = "Excellent"
            confidence = "Medium"
        
        logger.info(f"Prediction made: {quality_score:.2f} ({category})")
        
        return PredictionResponse(
            quality_score=round(float(quality_score), 2),
            quality_category=category,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )