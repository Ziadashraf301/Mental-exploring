"""
Production Inference Pipeline for Emotion Detection
Simple, clean, and ready for FastAPI integration
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import mlflow.tensorflow
import os
import json
from datetime import datetime
import logging
from config import settings


class EmotionDetectionPipeline:
    def __init__(self):
        # Load configuration
        self.CONFIG = settings

        # Setup logger
        logging.basicConfig(
            level=getattr(logging, self.CONFIG.EMOTION_LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.CONFIG.EMOTION_LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.LOGGER = logging.getLogger(__name__)

        # Initialize Emotion model and face detector placeholders
        self.MODEL = None
        self.FACE_DETECTOR = None

    # INITIALIZATION
    def initialize_pipeline(self):
        """
        Initialize the inference pipeline
        Load config, setup logger, load model and face detector
        """
        self.LOGGER.info("INITIALIZING EMOTION DETECTION INFERENCE PIPELINE")

        mlflow.set_tracking_uri(self.CONFIG.MLFLOW_TRACKING_URI)
        self.LOGGER.info(f"\n✓ MLflow Tracking URI set: {self.CONFIG.MLFLOW_TRACKING_URI}")

        # Load model
        model_uri = f"models:/{self.CONFIG.EMOTION_MODEL_NAME}/{self.CONFIG.EMOTION_MODEL_VERSION}"
        self.LOGGER.info(f"✓ Loading model from: {model_uri}")
        try:
            self.MODEL = mlflow.tensorflow.load_model(model_uri)
            self.LOGGER.info(f"✓ Model loaded successfully: {self.CONFIG.EMOTION_MODEL_NAME}")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to load model: {str(e)}")
            raise

        # Initialize face detector
        self.LOGGER.info("✓ Initializing MTCNN face detector...")
        try:
            self.FACE_DETECTOR = MTCNN()
            self.LOGGER.info("✓ Face detector initialized successfully")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to initialize face detector: {str(e)}")
            raise

        self.LOGGER.info("PIPELINE INITIALIZATION COMPLETE")

    # FACE DETECTION
    def detect_faces(self, image_path, min_confidence=None):
        if min_confidence is None:
            min_confidence = self.CONFIG.EMOTION_FACE_CONFIDENCE_THRESHOLD

        image = cv2.imread(image_path)
        if image is None:
            self.LOGGER.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Error: Could not load image at {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.LOGGER.info("Detecting faces...")
        faces = self.FACE_DETECTOR.detect_faces(image_rgb)

        results = []
        for face in faces:
            if face['confidence'] >= min_confidence:
                x, y, w, h = face['box']
                results.append({
                    "box": (x, y, w, h),
                    "confidence": float(face['confidence'])
                })

        self.LOGGER.info(f"Found {len(results)} face(s) with confidence >= {min_confidence}")
        return results

    # EMOTION RECOGNITION
    def predict_emotion(self, image_path, faces):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        emotions = []

        self.LOGGER.info("Predicting emotions...")

        for idx, face in enumerate(faces):
            x, y, w, h = face["box"]
            face_region = image_rgb[y:y+h, x:x+w]
            if face_region.size == 0:
                self.LOGGER.warning(f"Empty face region for face {idx + 1}, skipping...")
                continue

            target_size = self.CONFIG.EMOTION_IMAGE_SIZE
            resized = cv2.resize(face_region, target_size)

            if self.CONFIG.EMOTION_NORMALIZE:
                preprocessed = resized.astype(np.float32) / 255.0
            else:
                preprocessed = resized.astype(np.float32)

            preprocessed = np.expand_dims(preprocessed, axis=0)

            sad_prob = float(self.MODEL.predict(preprocessed, verbose=0)[0][0])
            happy_prob = 1.0 - sad_prob

            emotions.append({
                "face_id": idx + 1,
                "sad_probability": sad_prob,
                "happy_probability": happy_prob,
                "dominant_emotion": "sad" if sad_prob > 0.5 else "happy",
                "confidence": face["confidence"],
                "box": face["box"]
            })

            self.LOGGER.info(
                f"Face {idx + 1}: {emotions[-1]['dominant_emotion']} "
                f"(sad={sad_prob:.4f}, happy={happy_prob:.4f})"
            )

        return emotions

    # MAIN PROCESSING FUNCTION
    def process_image(self, image_path, min_face_confidence=None, save_result=False):
        self.LOGGER.info(f"PROCESSING IMAGE: {image_path}")

        try:
            faces = self.detect_faces(image_path, min_face_confidence)

            if not faces:
                self.LOGGER.warning("No faces detected in image")
                result = {
                    "success": True,
                    "image_path": image_path,
                    "timestamp": datetime.now().isoformat(),
                    "faces_detected": 0,
                    "results": [],
                    "message": "No faces detected"
                }
                return result

            emotions = self.predict_emotion(image_path, faces)

            result = {
                "success": True,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "faces_detected": len(faces),
                "faces_processed": len(emotions),
                "results": emotions,
                "model_info": {
                    "name": self.CONFIG.EMOTION_MODEL_NAME,
                    "version": self.CONFIG.EMOTION_MODEL_VERSION,
                    "stage": self.CONFIG.EMOTION_MODEL_STAGE
                }
            }

            if save_result or self.CONFIG.EMOTION_SAVE_RESULTS:
                self.save_results_to_json(result)

            self.LOGGER.info(f"PROCESSING COMPLETE: {len(emotions)} face(s) processed")
            return result

        except Exception as e:
            self.LOGGER.error(f"Error processing image: {str(e)}", exc_info=True)
            return {
                "success": False,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "faces_detected": 0,
                "results": [],
                "message": "Error during processing"
            }

    # UTILITY FUNCTIONS
    def save_results_to_json(self, result):

        os.makedirs(self.CONFIG.EMOTION_RESULTS_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.json"
        filepath = os.path.join(self.CONFIG.EMOTION_RESULTS_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        self.LOGGER.info(f"Results saved to: {filepath}")

    def get_model_info(self):
        return {
            "model_loaded": self.MODEL is not None,
            "model_name": self.CONFIG.EMOTION_MODEL_NAME,
            "model_version": self.CONFIG.EMOTION_MODEL_VERSION,
            "model_stage": self.CONFIG.EMOTION_MODEL_STAGE,
            "tracking_uri": self.CONFIG.MLFLOW_TRACKING_URI
        }

    def health_check(self):
        return {
            "status": "healthy" if (self.MODEL is not None and self.FACE_DETECTOR is not None) else "unhealthy",
            "model_loaded": self.MODEL is not None,
            "face_detector_loaded": self.FACE_DETECTOR is not None,
            "config_loaded": self.CONFIG is not None,
            "model_info": self.get_model_info() if self.CONFIG else None
        }
