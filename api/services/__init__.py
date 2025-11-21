"""
Services package initialization
"""

from .emotion_service.emotion_service import EmotionDetectionService, get_emotion_service
# from .depression_service.depression_service import DepressionDetectionService, get_depression_service
from .sentiment_service.sentiment_service import SentimentAnalysisService, get_sentiment_service

__all__ = [
    'EmotionDetectionService',
    'get_emotion_service',
    'SentimentAnalysisService',
    'get_sentiment_service'
]