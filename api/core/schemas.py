"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime
from typing import Optional, Dict, List


# USER SCHEMAS
class UserCreate(BaseModel):
    """Create user request"""
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    metadata: Optional[Dict] = None


class UserResponse(BaseModel):
    """User response"""
    user_id: str
    name: str
    email: Optional[EmailStr] = None
    created_at: datetime
    total_requests: int = 0


# DEPRESSION DETECTION SCHEMAS
class DepressionRequest(BaseModel):
    """Depression detection request"""
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v


class DepressionResponse(BaseModel):
    """Depression detection response"""
    prediction: str  # "Depressed" or "Not Depressed"
    confidence: float
    depression_probability: float
    not_depression_probability: float
    user_id: str
    prediction_id: str
    inference_time: float
    model_name: str
    model_version: str
    timestamp: datetime


# EMOTION DETECTION SCHEMAS
class FaceEmotion(BaseModel):
    """Single face emotion result"""
    face_id: int
    bounding_box: Dict[str, int]
    face_confidence: float
    emotions: Dict[str, float] 
    dominant_emotion: str


class EmotionResponse(BaseModel):
    """Emotion detection response"""
    success: bool
    faces_detected: int
    faces_processed: int
    results: List[FaceEmotion]
    user_id: str
    prediction_id: str
    inference_time: float
    model_version: str
    timestamp: datetime


# SENTIMENT ANALYSIS SCHEMAS
class SentimentRequest(BaseModel):
    """Sentiment analysis request"""
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v


class SentimentResponse(BaseModel):
    """Sentiment analysis response"""
    success: bool
    prediction: str  
    confidence: float
    positive_probability: float
    negative_probability: float
    user_id: str
    prediction_id: str
    inference_time: float
    model_name: str
    timestamp: datetime


# ANALYTICS SCHEMAS
class AnalyticsResponse(BaseModel):
    """Analytics response"""
    total_predictions: int
    unique_users: int
    avg_confidence: float
    avg_inference_time: float
    predictions_by_service: Dict[str, int]
    predictions_by_date: List[Dict]
    period_days: int
    timestamp: datetime


# HEALTH CHECK SCHEMAS
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, bool]
    database_connected: bool
    mlflow_connected: bool
    timestamp: datetime


# BATCH PREDICTION SCHEMAS
class BatchDepressionRequest(BaseModel):
    """Batch depression detection request"""
    texts: List[str] = Field(..., max_items=100)
    user_id: Optional[str] = None


class BatchSentimentRequest(BaseModel):
    """Batch sentiment analysis request"""
    texts: List[str] = Field(..., max_items=100)
    user_id: Optional[str] = None