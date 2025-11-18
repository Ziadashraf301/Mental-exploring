"""
Emotion Detection Service
Uses the EmotionDetectionPipeline class from Emotion_detection project
"""

import logging
from config import settings
from services.emotion_service.emotion_pipeline import EmotionDetectionPipeline  

logger = logging.getLogger(__name__)


class EmotionDetectionService:
    """
    Emotion detection service
    Wraps the EmotionDetectionPipeline class
    """

    def __init__(self):
        self.initialized = False
        self.config = settings
        self.model_info = None
        self.pipeline = EmotionDetectionPipeline()  # <-- pipeline class instance

    def initialize(self):
        """
        Initialize the emotion detection pipeline
        """
        if self.initialized:
            logger.info("Emotion detection service already initialized")
            return

        try:
            logger.info("Initializing emotion detection service...")

            # Initialize pipeline
            self.pipeline.initialize_pipeline()

            # Store model info
            self.model_info = self.pipeline.get_model_info()

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
            result = self.pipeline.process_image(
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
            return self.pipeline.health_check()
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


# Global singleton service instance
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
