"""
Services package initialization
"""

from .emotion_service import EmotionDetectionService, get_emotion_service
# from .depression_service import DepressionDetectionService, get_depression_service
# from .sentiment_service import SentimentAnalysisService, get_sentiment_service

__all__ = [
    'EmotionDetectionService',
    'get_emotion_service'
]