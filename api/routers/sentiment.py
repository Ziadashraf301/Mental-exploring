"""
SENTIMENT ANALYSIS API Router
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import os
import numpy as np
import time
import logging

from core.schemas import EmotionResponse, SentimentRequest, SentimentResponse
from core.database import Database
from services import get_sentiment_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])

# Dependency injection
def get_db():
    """Get database instance"""
    from main import db
    return db


@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    db: Database = Depends(get_db),
):
    """
    Detect sentiment in text

    - request: SentimentRequest(text, user_id, metadata)

    Returns sentiment analysis results
    """
    
    # Validate file type
    if not request.text or not isinstance(request.text, str):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid text input. Please provide a non-empty string."
        )

    try:
        start_time = time.time()
        
        # Handle user_id
        if not request.user_id:
            user_id = await db.create_user(name="Anonymous")
        else:
            user = await db.get_user(request.user_id)
            if user:
                user_id = user.id 
            else:
                user_id = await db.create_user(name="Anonymous")

        # Get sentiment service
        sentiment_service = get_sentiment_service()
        
        # Process text
        result = sentiment_service.predict(
            text=request.text
        )
        
        inference_time = time.time() - start_time
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Sentiment analysis failed")
            )

        # Save to database
        prediction_id = await db.save_prediction(
            user_id=user_id,
            service_type="sentiment",
            input_text=request.text,
            input_length=len(request.text),
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities={
                "negative": result["probability_negative"],
                "positive": result["probability_positive"]
            },
            model_name=result["model_info"]["name"],
            model_version=result["model_info"]["version"],
            inference_time=inference_time,
        )

        # Return response
        return SentimentResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            positive_probability=result["probability_positive"],
            negative_probability=result["probability_negative"],
            user_id=user_id,
            prediction_id=prediction_id,
            inference_time=inference_time,
            model_name=result["model_info"]["name"],
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:    
        logger.error(f"Sentiment prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )


@router.get("/health")
async def sentiment_health_check():
    """
    Check  service health
    
    Returns service status and model information
    """
    try:
        sentiment_service = get_sentiment_service()
        health = sentiment_service.health_check()
        return health
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/model/info")
async def get_emotion_model_info():
    """
    Get emotion detection model information
    
    Returns model details and configuration
    """
    try:
        sentiment_service = get_sentiment_service()
        info = sentiment_service.get_info()
        return info
    except Exception as e:
        return {
            "error": str(e)
        }