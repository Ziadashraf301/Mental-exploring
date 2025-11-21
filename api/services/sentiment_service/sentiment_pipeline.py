"""
Production Inference Pipeline for Sentiment Analysis
Simple, clean, and ready for FastAPI integration
"""

import mlflow.sklearn
import os
import json
from datetime import datetime
import logging
from config import settings
import re
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


class SentimentAnalysisPipeline:
    def __init__(self):
        # Load configuration
        self.CONFIG = settings

        # Setup logger
        logging.basicConfig(
            level=getattr(logging, self.CONFIG.SENTIMENT_LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.CONFIG.SENTIMENT_LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.LOGGER = logging.getLogger(__name__)

        # Initialize Sentiment model placeholder
        self.MODEL = None
        self.VECTORIZER = None

    def initialize_pipeline(self):
        """
        Initialize the inference pipeline
        Load config, setup logger, load model
        """
        self.LOGGER.info("INITIALIZING SENTIMENT ANALYSIS INFERENCE PIPELINE")

        mlflow.set_tracking_uri(self.CONFIG.MLFLOW_TRACKING_URI)
        self.LOGGER.info(f"\n✓ MLflow Tracking URI set: {self.CONFIG.MLFLOW_TRACKING_URI}")

        # Load model
        model_uri = f"models:/{self.CONFIG.SENTIMENT_MODEL_NAME}/{self.CONFIG.SENTIMENT_MODEL_VERSION}"
        self.LOGGER.info(f"✓ Loading model from: {model_uri}")
        try:
            self.MODEL = mlflow.sklearn.load_model(model_uri)
            self.LOGGER.info(f"✓ Model loaded successfully: {self.CONFIG.SENTIMENT_MODEL_NAME}")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to load model: {str(e)}")
            raise

        vactorizer_uri = f"models:/{self.CONFIG.SENTIMENT_VACTORIZER_MODEL}/{self.CONFIG.SENTIMENT_VACTORIZER_MODEL_VERSION}"
        self.LOGGER.info(f"✓ Loading vectorizer from: {vactorizer_uri}")
        try:
            self.VECTORIZER = mlflow.sklearn.load_model(vactorizer_uri)
            self.LOGGER.info(f"✓ Vectorizer loaded successfully: {self.CONFIG.SENTIMENT_VACTORIZER_MODEL}")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to load vectorizer: {str(e)}")
            raise

        self.LOGGER.info("PIPELINE INITIALIZATION COMPLETE")

    def preprocess_text(self, text):

        try:
            # Create Lemmatizer
            wordLemm = WordNetLemmatizer()
            
            # Defining regex patterns.
            urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
            userPattern       = '@[^\s]+'
            alphaPattern      = "[^a-zA-Z0-9]"
            sequencePattern   = r"(.)\1\1+"
            seqReplacePattern = r"\1\1"

            # Defining dictionary containing all emojis with their meanings.
            emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
                ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
                ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
                ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

            text = text.lower()
                
            # Replace all URls with 'URL'
            text = re.sub(urlPattern,'',text)

            # Replace all emojis.
            for emoji in emojis.keys():
                text = text.replace(emoji,emojis[emoji])        

            # Replace @USERNAME to 'USER'.
            text = re.sub(userPattern,'', text)

            # Replace all non alphabets.
            text = re.sub(alphaPattern, " ", text)

            # Replace 3 or more consecutive letters by 2 letter.
            text = re.sub(sequencePattern, seqReplacePattern, text)

            sentence = ''
            for word in text.split():
                if len(word)>1:
                    word = wordLemm.lemmatize(word)
                    sentence += (word+' ')
            
            self.LOGGER.info(f"✓ Text preprocessing complete")
        
        except Exception as e:
            self.LOGGER.error(f"✗ Text preprocessing failed: {str(e)}")
            raise
            
        return sentence

    def predict_sentiment(self, text, save_result=False):
        try:
            if not text:
                self.LOGGER.warning("No text provided for sentiment analysis")
                result = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "results": [],
                    "message": "No text provided for sentiment analysis"
                }
                return result
         
            # preprocess text
            text = self.preprocess_text(text)

            # vactorize text
            text_vactor = self.VECTORIZER.transform([text])

            # make prediction
            sentiment = self.MODEL.predict(text_vactor)[0]
            propabilty = self.MODEL.predict_proba(text_vactor)

            # get probabilities
            probabilty_negative = propabilty[:,0][0]
            probabilty_positive = propabilty[:,1][0]

            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "prediction": "Positive" if sentiment == 1 else "Negative",
                "probability_negative": float(f"{probabilty_negative:.4f}"),
                "probability_positive": float(f"{probabilty_positive:.4f}"),
                "confidence": float(f"{max(probabilty_negative, probabilty_positive):.4f}"),
                "model_info": {
                    "name": self.CONFIG.SENTIMENT_MODEL_NAME,
                    "version": self.CONFIG.SENTIMENT_MODEL_VERSION,
                    "stage": self.CONFIG.SENTIMENT_MODEL_STAGE,
                    "vectorizer_name": self.CONFIG.SENTIMENT_VACTORIZER_MODEL,
                    "vectorizer_version": self.CONFIG.SENTIMENT_VACTORIZER_MODEL_VERSION
                }
            }

            if save_result or self.CONFIG.SENTIMENT_SAVE_RESULTS:
                self.save_results_to_json(result)

            self.LOGGER.info("✓ Sentiment prediction complete")
            return result
        
        except Exception as e:
            self.LOGGER.error(f"✗ Error during sentiment prediction: {str(e)}", exc_info=True)
            return {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "results": [],
                "message": "Error during processing"

            }

    def save_results_to_json(self, result):
        os.makedirs(self.CONFIG.SENTIMENT_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.json"
        filepath = os.path.join(self.CONFIG.SENTIMENT_RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        self.LOGGER.info(f"Results saved to: {filepath}")

    def get_model_info(self):
        return {
            "model_loaded": self.MODEL is not None,
            "model_name": self.CONFIG.SENTIMENT_MODEL_NAME,
            "model_version": self.CONFIG.SENTIMENT_MODEL_VERSION,
            "model_stage": self.CONFIG.SENTIMENT_MODEL_STAGE,
            "tracking_uri": self.CONFIG.MLFLOW_TRACKING_URI
        }

    def health_check(self):
        return {
            "status": "healthy" if (self.MODEL is not None) else "unhealthy",
            "model_loaded": self.MODEL is not None,
            "config_loaded": self.CONFIG is not None,
            "model_info": self.get_model_info() if self.CONFIG else None
        }