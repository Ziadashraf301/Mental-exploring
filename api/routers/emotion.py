"""
Emotion Detection API Router
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, BackgroundTasks
from datetime import datetime
import tempfile
import os
import cv2
import numpy as np
import time
import logging

from core.schemas import EmotionRequest, EmotionResponse, FaceEmotion
from core.database import Database
from services.emotion_service import get_emotion_service
from utils.mlflow_tracker import MLflowTracker
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion", tags=["Emotion Detection"])


# Dependency injection
def get_db():
    """Get database instance"""
    from main import db
    return db


def get_tracker():
    """Get MLflow tracker instance"""
    from main import mlflow_tracker
    return mlflow_tracker


@router.post("/predict", response_model=EmotionResponse)
async def predict_emotion(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    user_id: str = Query(None, description="User identifier"),
    min_face_confidence: float = Query(None, ge=0.0, le=1.0, description="Minimum face detection confidence"),
    db: Database = Depends(get_db),
    tracker: MLflowTracker = Depends(get_tracker)
):
    """
    Detect emotions in uploaded image
    
    - **file**: Image file containing faces
    - **user_id**: Optional user identifier
    - **min_face_confidence**: Minimum confidence threshold for face detection
    
    Returns emotion predictions for all detected faces
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # Create temporary file
    temp_path = None
    try:
        start_time = time.time()
        
        # Handle user_id
        if not user_id:
            user_id = await db.create_user(name="Anonymous")
        else:
            user = await db.get_user(user_id)
            if not user:
                user_id = await db.create_user(name="Anonymous")
        
        # Read uploaded file
        contents = await file.read()
        
        # Decode image to verify it's valid
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please upload a valid image file."
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Get emotion service
        emotion_service = get_emotion_service()
        
        # Process image
        result = emotion_service.predict(
            image_path=temp_path,
            min_face_confidence=min_face_confidence
        )
        
        inference_time = time.time() - start_time
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Emotion detection failed")
            )
        
        # Prepare response
        face_emotions = []
        for face_result in result["results"]:
            face_emotions.append(FaceEmotion(
                face_id=face_result["face_id"],
                bounding_box=face_result["box"] if isinstance(face_result["box"], dict) else {
                    "x": face_result["box"][0],
                    "y": face_result["box"][1],
                    "width": face_result["box"][2],
                    "height": face_result["box"][3]
                },
                face_confidence=face_result["confidence"],
                emotions={
                    "sad": face_result["sad_probability"],
                    "happy": face_result["happy_probability"]
                },
                dominant_emotion=face_result["dominant_emotion"]
            ))
        
        # Get average confidence
        avg_confidence = sum(f.face_confidence for f in face_emotions) / len(face_emotions) if face_emotions else 0.0
        
        # Save to database (background task)
        prediction_id = None
        if face_emotions:
            prediction_id = await db.save_prediction(
                user_id=user_id,
                service_type="emotion",
                prediction=face_emotions[0].dominant_emotion,  # Primary face emotion
                confidence=avg_confidence,
                model_name=result["model_info"]["name"],
                model_version=result["model_info"]["version"],
                inference_time=inference_time,
                input_image_path=file.filename,
                probabilities={
                    "faces": [
                        {
                            "face_id": f.face_id,
                            "emotions": f.emotions,
                            "dominant": f.dominant_emotion
                        }
                        for f in face_emotions
                    ]
                }
            )
            
            # Log to MLflow (background)
            background_tasks.add_task(
                tracker.log_prediction,
                service_type="emotion",
                prediction=face_emotions[0].dominant_emotion,
                confidence=avg_confidence,
                inference_time=inference_time,
                metadata={
                    "faces_detected": result["faces_detected"],
                    "faces_processed": result["faces_processed"]
                }
            )
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # Return response
        return EmotionResponse(
            success=True,
            faces_detected=result["faces_detected"],
            faces_processed=result["faces_processed"],
            results=face_emotions,
            user_id=user_id,
            prediction_id=prediction_id or "N/A",
            inference_time=inference_time,
            model_version=result["model_info"]["version"],
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Emotion prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.get("/health")
async def emotion_health_check():
    """
    Check emotion detection service health
    
    Returns service status and model information
    """
    try:
        emotion_service = get_emotion_service()
        health = emotion_service.health_check()
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
        emotion_service = get_emotion_service()
        info = emotion_service.get_info()
        return info
    except Exception as e:
        return {
            "error": str(e)
        }