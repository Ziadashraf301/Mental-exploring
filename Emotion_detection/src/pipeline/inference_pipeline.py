"""
Production Inference Pipeline for Emotion Detection
Simple, clean, and ready for FastAPI integration
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import mlflow.tensorflow
import os
import json
from datetime import datetime

from src.config.inference_config_loader import get_inference_config, reload_inference_config
from src.logger.inference_logger import setup_inference_logger


# ===========================================
# GLOBAL VARIABLES
# ===========================================
CONFIG = None
LOGGER = None
MODEL = None
FACE_DETECTOR = None


# ===========================================
# INITIALIZATION
# ===========================================
def initialize_pipeline(config_path="inference_config.yaml"):
    """
    Initialize the inference pipeline
    Load config, setup logger, load model and face detector
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    tuple
        (config, logger, model, face_detector)
    """
    global CONFIG, LOGGER, MODEL, FACE_DETECTOR
    
    # Load configuration
    CONFIG = reload_inference_config(config_path)
    
    # Setup logger
    LOGGER = setup_inference_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("=" * 70)
    LOGGER.info("INITIALIZING EMOTION DETECTION INFERENCE PIPELINE")
    LOGGER.info("=" * 70)
    
    # Print configuration
    CONFIG.print_config()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    LOGGER.info(f"\n✓ MLflow Tracking URI set: {CONFIG.mlflow_tracking_uri}")
    
    # Load model
    LOGGER.info(f"✓ Loading model from: {CONFIG.model_uri}")
    try:
        MODEL = mlflow.tensorflow.load_model(CONFIG.model_uri)
        LOGGER.info(f"✓ Model loaded successfully: {CONFIG.model_name}")
    except Exception as e:
        LOGGER.error(f"✗ Failed to load model: {str(e)}")
        raise
    
    # Initialize face detector
    LOGGER.info("✓ Initializing MTCNN face detector...")
    try:
        FACE_DETECTOR = MTCNN()
        LOGGER.info("✓ Face detector initialized successfully")
    except Exception as e:
        LOGGER.error(f"✗ Failed to initialize face detector: {str(e)}")
        raise
    
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("PIPELINE INITIALIZATION COMPLETE")
    LOGGER.info("=" * 70 + "\n")
    
    return CONFIG, LOGGER, MODEL, FACE_DETECTOR


# ===========================================
# FACE DETECTION
# ===========================================
def detect_faces(image_path, min_confidence=None):
    """
    Detect faces with MTCNN and return bounding boxes + confidence
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    min_confidence : float, optional
        Minimum confidence threshold. If None, uses config value
    
    Returns:
    --------
    list
        List of detected faces with bounding boxes and confidence
    """
    global CONFIG, LOGGER, FACE_DETECTOR
    
    if min_confidence is None:
        min_confidence = CONFIG.face_confidence_threshold
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        LOGGER.error(f"Failed to load image: {image_path}")
        raise ValueError(f"Error: Could not load image at {image_path}")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    LOGGER.info("Detecting faces...")
    faces = FACE_DETECTOR.detect_faces(image_rgb)
    
    # Filter by confidence
    results = []
    for face in faces:
        if face['confidence'] >= min_confidence:
            x, y, w, h = face['box']
            results.append({
                "box": (x, y, w, h),
                "confidence": float(face['confidence'])
            })
    
    LOGGER.info(f"Found {len(results)} face(s) with confidence >= {min_confidence}")
    
    return results


# ===========================================
# EMOTION RECOGNITION
# ===========================================
def predict_emotion(image_path, faces):
    """
    Takes detected faces and predicts emotion for each
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    faces : list
        List of detected faces from detect_faces()
    
    Returns:
    --------
    list
        List of emotion predictions for each face
    """
    global CONFIG, LOGGER, MODEL
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    emotions = []
    
    LOGGER.info("Predicting emotions...")
    
    for idx, face in enumerate(faces):
        x, y, w, h = face["box"]
        
        # Crop face
        face_region = image_rgb[y:y+h, x:x+w]
        if face_region.size == 0:
            LOGGER.warning(f"Empty face region for face {idx + 1}, skipping...")
            continue
        
        # Resize to model input size
        target_size = CONFIG.image_size
        resized = cv2.resize(face_region, target_size)
        
        # Normalize if configured
        if CONFIG.normalize:
            preprocessed = resized.astype(np.float32) / 255.0
        else:
            preprocessed = resized.astype(np.float32)
        
        # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Predict
        sad_prob = float(MODEL.predict(preprocessed, verbose=0)[0][0])
        happy_prob = 1.0 - sad_prob
        
        emotions.append({
            "face_id": idx + 1,
            "sad_probability": sad_prob,
            "happy_probability": happy_prob,
            "dominant_emotion": "sad" if sad_prob > 0.5 else "happy",
            "confidence": face["confidence"],
            "box": face["box"]
        })
        
        LOGGER.info(
            f"Face {idx + 1}: {emotions[-1]['dominant_emotion']} "
            f"(sad={sad_prob:.4f}, happy={happy_prob:.4f})"
        )
    
    return emotions


# ===========================================
# MAIN PROCESSING FUNCTION
# ===========================================
def process_image(image_path, min_face_confidence=None, save_result=False):
    """
    Complete processing pipeline: detect faces and predict emotions
    This is the main entry point for FastAPI
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    min_face_confidence : float, optional
        Minimum face detection confidence
    save_result : bool
        Whether to save results to JSON file
    
    Returns:
    --------
    dict
        Processing results with all detected faces and emotions
    """
    global CONFIG, LOGGER
    
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info(f"PROCESSING IMAGE: {image_path}")
    LOGGER.info(f"{'='*70}")
    
    try:
        # Step 1: Face Detection
        faces = detect_faces(image_path, min_face_confidence)
        
        if not faces:
            LOGGER.warning("No faces detected in image")
            result = {
                "success": True,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "faces_detected": 0,
                "results": [],
                "message": "No faces detected"
            }
            return result
        
        # Step 2: Emotion Prediction
        emotions = predict_emotion(image_path, faces)
        
        # Prepare result
        result = {
            "success": True,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "faces_detected": len(faces),
            "faces_processed": len(emotions),
            "results": emotions,
            "model_info": {
                "name": CONFIG.model_name,
                "version": CONFIG.model_version,
                "stage": CONFIG.model_stage
            }
        }
        
        # Save result if configured
        if save_result or CONFIG.save_results:
            save_results_to_json(result)
        
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info(f"PROCESSING COMPLETE: {len(emotions)} face(s) processed")
        LOGGER.info(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        LOGGER.error(f"Error processing image: {str(e)}", exc_info=True)
        return {
            "success": False,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "faces_detected": 0,
            "results": []
        }


# ===========================================
# UTILITY FUNCTIONS
# ===========================================
def save_results_to_json(result):
    """
    Save processing results to JSON file
    
    Parameters:
    -----------
    result : dict
        Processing results
    """
    global CONFIG, LOGGER
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.json"
    filepath = os.path.join(CONFIG.results_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    LOGGER.info(f"Results saved to: {filepath}")


def get_model_info():
    """
    Get current model information
    
    Returns:
    --------
    dict
        Model information
    """
    global CONFIG, MODEL
    
    return {
        "model_loaded": MODEL is not None,
        "model_name": CONFIG.model_name,
        "model_version": CONFIG.model_version,
        "model_stage": CONFIG.model_stage,
        "model_uri": CONFIG.model_uri,
        "tracking_uri": CONFIG.mlflow_tracking_uri
    }


def health_check():
    """
    Check if pipeline is healthy and ready
    
    Returns:
    --------
    dict
        Health status
    """
    global CONFIG, LOGGER, MODEL, FACE_DETECTOR
    
    return {
        "status": "healthy" if (MODEL is not None and FACE_DETECTOR is not None) else "unhealthy",
        "model_loaded": MODEL is not None,
        "face_detector_loaded": FACE_DETECTOR is not None,
        "config_loaded": CONFIG is not None,
        "model_info": get_model_info() if CONFIG else None
    }


# ===========================================
# MAIN FUNCTION (CLI)
# ===========================================
def main():
    """
    Main function for command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Emotion Detection Inference Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="inference_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=None,
        help="Minimum face detection confidence (0.0-1.0)"
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    initialize_pipeline(args.config)
    
    # Process image
    result = process_image(
        image_path=args.image,
        min_face_confidence=args.face_confidence,
        save_result=args.save_result
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if result["success"]:
        print(f"\n✓ Successfully processed image")
        print(f"  Image: {result['image_path']}")
        print(f"  Faces Detected: {result['faces_detected']}")
        print(f"  Faces Processed: {result['faces_processed']}")
        
        for face_result in result["results"]:
            x, y, w, h = face_result['box']
            print(f"\n  Face {face_result['face_id']}:")
            print(f"    • Bounding Box: (x={x}, y={y}, w={w}, h={h})")
            print(f"    • Detection Confidence: {face_result['confidence']:.3f}")
            print(f"    • Dominant Emotion: {face_result['dominant_emotion'].upper()}")
            print(f"    • Sad Probability: {face_result['sad_probability']:.4f}")
            print(f"    • Happy Probability: {face_result['happy_probability']:.4f}")
    else:
        print(f"\n✗ Processing failed: {result['error']}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()