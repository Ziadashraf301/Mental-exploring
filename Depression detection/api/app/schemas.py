from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime
from typing import Optional, Dict, List
from langdetect import detect, LangDetectException

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    depression_probability: float
    not_depression_probability: float
    user_id: str
    timestamp: datetime
    inference_time: float
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Depressed",
                "confidence": 0.9459,
                "depression_probability": 0.9459,
                "not_depression_probability": 0.0541,
                "user_id": "user-123",
                "timestamp": "2025-01-15T10:30:00",
                "inference_time": 0.045,
                "model_version": "1.0"
            }
        }

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    metadata: Optional[Dict] = None

class UserResponse(BaseModel):
    user_id: str
    name: str
    email: Optional[EmailStr] = None
    created_at: datetime

class AnalyticsResponse(BaseModel):
    total_predictions: int
    unique_users: int
    depression_rate: float
    avg_confidence: float
    avg_inference_time: float
    predictions_by_date: List[Dict]
    period_days: int
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: datetime