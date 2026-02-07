# /src/config.py

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model settings
    MODEL_NAME: str = "wine_quality_rf"
    MODEL_VERSION: str = "v1"
    
    # Training parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_ESTIMATORS: int = 100
    MAX_DEPTH: int = 10
    
    # API settings
    API_TITLE: str = "Wine Quality Prediction API"
    API_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"

settings = Settings()