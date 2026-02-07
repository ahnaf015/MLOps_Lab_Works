# /src/data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():

    try:
        # Using wine quality dataset from sklearn
        from sklearn.datasets import fetch_openml
        
        logger.info("Loading wine quality dataset...")
        wine_data = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
        
        X = wine_data.data
        y = wine_data.target.astype(float)
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(X, y):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, test_size=None, random_state=None):

    test_size = test_size or settings.TEST_SIZE
    random_state = random_state or settings.RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test

def get_feature_names():
    """Return the feature names for the wine quality dataset"""
    return [
        'fixed_acidity',
        'volatile_acidity', 
        'citric_acid',
        'residual_sugar',
        'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]