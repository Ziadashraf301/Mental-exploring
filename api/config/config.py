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
    DATABASE_URL: str = "sqlite+aiosqlite:///./database/mental_health_api.db"
    # For PostgreSQL: "postgresql+asyncpg://user:password@localhost/dbname"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "mental_health_api"
    
    # Depression Detection Model
    DEPRESSION_MODEL_NAME: str = "ziadashraf98765/roberta-depression-detection-lora-merged"
    DEPRESSION_MODEL_VERSION: str = "1.0"
    
    # Sentiment Analysis Model
    SENTIMENT_MODEL_NAME: str = "Sentiment_analysisLogisticRegression_Model"
    SENTIMENT_MODEL_VERSION: str = "1.0"
    SENTIMENT_MODEL_STAGE: str = "Production"
    SENTIMENT_VACTORIZER_MODEL: str = "TFIDF_Vectorizer_Sentiment"
    SENTIMENT_VACTORIZER_MODEL_VERSION: str = "4.0"
    SENTIMENT_SAVE_RESULTS: bool = True
    SENTIMENT_RESULTS_DIR: str = "results"
    SENTIMENT_LOG_FILE: str = "logs/sentiment_service_inference.log"
    SENTIMENT_LOG_LEVEL: str = "INFO"

    # Emotion Detection Model
    EMOTION_MODEL_NAME: str = "CNN_EmotionDetection"
    EMOTION_MODEL_VERSION: str = "1"
    EMOTION_MODEL_STAGE: str = None  # "Production" or None
    EMOTION_FACE_CONFIDENCE_THRESHOLD: float = 0.9
    EMOTION_IMAGE_SIZE: List[int] = [48, 48]
    EMOTION_NORMALIZE: bool = True
    EMOTION_SAVE_RESULTS: bool = False
    EMOTION_RESULTS_DIR: str = "results"
    EMOTION_LOG_FILE: str = "logs/emotion_detection_inference.log"
    EMOTION_LOG_LEVEL: str = "INFO"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # Security
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""

    # AWS Settings
    AWS_ACCESS_KEY_ID: str = None
    AWS_SECRET_ACCESS_KEY: str = None
    AWS_DEFAULT_REGION: str = None

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set AWS credentials in environment automatically
        if self.AWS_ACCESS_KEY_ID:
            os.environ["AWS_ACCESS_KEY_ID"] = self.AWS_ACCESS_KEY_ID
        if self.AWS_SECRET_ACCESS_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.AWS_SECRET_ACCESS_KEY
        if self.AWS_DEFAULT_REGION:
            os.environ["AWS_DEFAULT_REGION"] = self.AWS_DEFAULT_REGION

settings = Settings()