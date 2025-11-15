"""
API Configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Mental Health Detection API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Unified API for Depression, Emotion, and Sentiment Detection"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./mental_health_api.db"
    # For PostgreSQL: "postgresql+asyncpg://user:password@localhost/dbname"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "mental_health_api"
    
    # Depression Detection Model
    DEPRESSION_MODEL_NAME: str = "ziadashraf98765/roberta-depression-detection-lora-merged"
    DEPRESSION_MODEL_VERSION: str = "1.0"
    
    # Emotion Detection Model
    EMOTION_MODEL_NAME: str = "CNN_EmotionDetection"
    EMOTION_MODEL_VERSION: str = "1"
    EMOTION_MODEL_STAGE: str = None  # or "Production"
    EMOTION_CONFIG_PATH: str = "../Emotion_detection/src/config/inference_config.yaml"
    
    # Sentiment Analysis Model
    SENTIMENT_MODEL_PATH: str = "../Sentiment_analysis/models"
    SENTIMENT_VECTORIZER_PATH: str = "../Sentiment_analysis/models/vectoriser.pkl"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # Security (optional)
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()