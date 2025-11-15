from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Model settings
    MODEL_NAME: str = "your-username/depression-detection-bert"
    MODEL_VERSION: str = "1.0"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./depression_detection.db"
    # For PostgreSQL: "postgresql+asyncpg://user:password@localhost/dbname"
    
    # Weights & Biases settings
    WANDB_PROJECT: str = "depression-detection-prod"
    WANDB_API_KEY: str = ""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()