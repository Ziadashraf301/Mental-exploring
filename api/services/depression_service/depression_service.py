"""
Depression Detection Service
Uses the DepressionDetectionPipeline class with BERT from Hugging Face
"""

import logging
from config import settings
from services.depression_service.depression_pipeline import DepressionDetectionPipeline  

logger = logging.getLogger(__name__)

class DepressionDetectionService:
    """
    Depression detection service using BERT
    Wraps the DepressionDetectionPipeline class
    """

    def __init__(self):
        self.initialized = False
        self.config = settings
        self.model_info = None
        self.pipeline = DepressionDetectionPipeline()

    
    def initialize(self):
        """
        Initialize the depression detection pipeline
        """
        if self.initialized:
            logger.info("Depression detection service already initialized")
            return

        try:
            logger.info("Initializing depression detection service...")
            
            # Initialize pipeline (loads BERT model from Hugging Face)
            self.pipeline.initialize_pipeline()

            # Store model info
            self.model_info = self.pipeline.get_model_info()

            self.initialized = True
            logger.info("✓ Depression detection service initialized successfully")
            logger.info(f"✓ Model: {self.model_info['model_name']}")
            logger.info(f"✓ Device: {self.model_info['device']}")

        except Exception as e:
            logger.error(f"Failed to initialize depression detection service: {str(e)}")
            raise

    def predict(self, text: str) -> dict:
        """
        Predict depression indicators in text
    
        Parameters:
        -----------
        text : str
            Text to analyze for depression indicators

        Returns:
        --------
        dict
            Prediction results with probabilities
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            result = self.pipeline.predict_depression(
                text=text,
                save_result=self.config.DEPRESSION_SAVE_RESULTS
            )
            return result

        except Exception as e:
            logger.error(f"Depression prediction failed: {str(e)}")
            raise

    def health_check(self) -> dict:
        """
        Check service health

        Returns:
        --------
        dict
            Health status including model and device info
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
            Model information including architecture and device
        """
        if not self.initialized:
            return {"error": "Service not initialized"}

        return self.model_info


# Global singleton service instance
_depression_service = None


def get_depression_service() -> DepressionDetectionService:
    """
    Get depression detection service instance (Singleton)

    Returns:
    --------
    DepressionDetectionService
        Service instance
    """
    global _depression_service
    if _depression_service is None:
        _depression_service = DepressionDetectionService()
    return _depression_service