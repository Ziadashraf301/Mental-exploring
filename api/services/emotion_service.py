"""
Emotion Detection Service
Uses the existing inference pipeline from Emotion_detection project
"""

import sys
import os
from pathlib import Path
import logging

# Add Emotion_detection to Python path
emotion_path = Path(__file__).parent.parent.parent / "Emotion_detection"
sys.path.insert(0, str(emotion_path))

from src import (
    initialize_pipeline,
    process_image,
    emotion_health_check,
    get_model_info
)

from src import get_inference_config

logger = logging.getLogger(__name__)


class EmotionDetectionService:
    """
    Emotion detection service
    Wraps the existing inference pipeline
    """
    
    def __init__(self):
        self.initialized = False
        self.config = None
        self.model_info = None
    
    def initialize(self, config_path: str = None):
        """
        Initialize the emotion detection pipeline
        
        Parameters:
        -----------
        config_path : str, optional
            Path to inference config file
        """
        if self.initialized:
            logger.info("Emotion detection service already initialized")
            return
        
        try:
            # Determine config path
            if config_path is None:
                config_path = emotion_path / "config" / "inference_config.yaml"
            
            logger.info(f"Initializing emotion detection service with config: {config_path}")
            
            # Initialize pipeline
            initialize_pipeline(str(config_path))
            
            # Store config
            self.config = get_inference_config(str(config_path))
            
            # Get model info
            self.model_info = get_model_info()
            
            self.initialized = True
            logger.info("✓ Emotion detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion detection service: {str(e)}")
            raise
    
    def predict(self, image_path: str, min_face_confidence: float = None) -> dict:
        """
        Predict emotions in image
        
        Parameters:
        -----------
        image_path : str
            Path to image file
        min_face_confidence : float, optional
            Minimum face detection confidence
        
        Returns:
        --------
        dict
            Prediction results
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            # Process image using existing pipeline
            result = process_image(
                image_path=image_path,
                min_face_confidence=min_face_confidence,
                save_result=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {str(e)}")
            raise
    
    def health_check(self) -> dict:
        """
        Check service health
        
        Returns:
        --------
        dict
            Health status
        """
        if not self.initialized:
            return {
                "status": "unhealthy",
                "message": "Service not initialized"
            }
        
        try:
            return emotion_health_check()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_info(self) -> dict:
        """
        Get model information
        
        Returns:
        --------
        dict
            Model information
        """
        if not self.initialized:
            return {"error": "Service not initialized"}
        
        return self.model_info


# Global service instance
_emotion_service = None


def get_emotion_service() -> EmotionDetectionService:
    """
    Get emotion detection service instance (Singleton)
    
    Returns:
    --------
    EmotionDetectionService
        Service instance
    """
    global _emotion_service
    if _emotion_service is None:
        _emotion_service = EmotionDetectionService()
    return _emotion_service