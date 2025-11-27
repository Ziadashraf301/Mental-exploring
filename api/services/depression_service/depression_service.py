"""
Depression Detection Service
Uses the DepressionDetectionPipeline class from Depression_detection project
"""

import logging
from config import settings
from services.depression_service.depression_pipeline import DepressionDetectionPipeline  

logger = logging.getLogger(__name__)

class DepressionDetectionService:
    """
    Sentiment analysis service
    Wraps the SentimentAnalysisPipeline class
    """

    def __init__(self):
        self.initialized = False
        self.config = settings
        self.model_info = None
        self.pipeline = DepressionDetectionPipeline()

    
    def initialize(self):
        """
        Initialize the sentiment analysis pipeline
        """
        if self.initialized:
            logger.info("Sentiment analysis service already initialized")
            return

        try:
            logger.info("Initializing sentiment analysis service...")
            
            # Initialize pipeline
            self.pipeline.initialize_pipeline()

            # Store model info
            self.model_info = self.pipeline.get_model_info()

            self.initialized = True
            logger.info("✓ Sentiment analysis service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sentiment analysis service: {str(e)}")
            raise

    def predict(self, text: str) -> dict:
        """
        Predict sentiment in text
    
        Parameters:
        -----------
        text : str
            Text to analyze

        Returns:
        --------
        dict
            Prediction results
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            result = self.pipeline.predict_sentiment(
                text=text,
                save_result=self.config.SENTIMENT_SAVE_RESULTS
            )
            return result

        except Exception as e:
            logger.error(f"Sentiment prediction failed: {str(e)}")
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


def get_sentiment_service() -> SentimentAnalysisService:
    """
    Get sentiment analysis service instance (Singleton)

    Returns:
    --------
    SentimentAnalysisService
        Service instance
    """
    global _emotion_service
    if _emotion_service is None:
        _emotion_service = SentimentAnalysisService()
    return _emotion_service
