"""
MLflow Tracking Utilities
"""

import mlflow
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow tracking wrapper"""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLflow tracker
        
        Parameters:
        -----------
        tracking_uri : str
            MLflow tracking server URI
        experiment_name : str
            Experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"✓ MLflow tracking initialized: {tracking_uri}")
            logger.info(f"✓ Experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {str(e)}")
    
    def log_prediction(
        self,
        service_type: str,
        prediction: str,
        confidence: float,
        inference_time: float,
        metadata: Optional[Dict] = None
    ):
        """
        Log prediction to MLflow
        
        Parameters:
        -----------
        service_type : str
            Type of service (depression, emotion, sentiment)
        prediction : str
            Prediction result
        confidence : float
            Prediction confidence
        inference_time : float
            Inference time in seconds
        metadata : Dict, optional
            Additional metadata
        """
        try:
            with mlflow.start_run(run_name=f"{service_type}_prediction") as run:
                # Log parameters
                mlflow.log_param("service_type", service_type)
                mlflow.log_param("prediction", prediction)
                
                # Log metrics
                mlflow.log_metric("confidence", confidence)
                mlflow.log_metric("inference_time", inference_time)
                
                # Log metadata
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                        else:
                            mlflow.log_param(key, str(value))
                
        except Exception as e:
            logger.error(f"Failed to log prediction to MLflow: {str(e)}")
    
    def log_batch(
        self,
        service_type: str,
        total_predictions: int,
        avg_confidence: float,
        avg_inference_time: float,
        success_count: int,
        failure_count: int
    ):
        """
        Log batch prediction metrics
        
        Parameters:
        -----------
        service_type : str
            Type of service
        total_predictions : int
            Total number of predictions
        avg_confidence : float
            Average confidence
        avg_inference_time : float
            Average inference time
        success_count : int
            Number of successful predictions
        failure_count : int
            Number of failed predictions
        """
        try:
            with mlflow.start_run(run_name=f"{service_type}_batch") as run:
                mlflow.log_param("service_type", service_type)
                mlflow.log_metric("total_predictions", total_predictions)
                mlflow.log_metric("avg_confidence", avg_confidence)
                mlflow.log_metric("avg_inference_time", avg_inference_time)
                mlflow.log_metric("success_count", success_count)
                mlflow.log_metric("failure_count", failure_count)
                mlflow.log_metric("success_rate", success_count / total_predictions if total_predictions > 0 else 0)
                
        except Exception as e:
            logger.error(f"Failed to log batch to MLflow: {str(e)}")
    
    def log_error(self, service_type: str, error_type: str, error_message: str):
        """
        Log error to MLflow
        
        Parameters:
        -----------
        service_type : str
            Type of service
        error_type : str
            Type of error
        error_message : str
            Error message
        """
        try:
            with mlflow.start_run(run_name=f"{service_type}_error") as run:
                mlflow.log_param("service_type", service_type)
                mlflow.log_param("error_type", error_type)
                mlflow.log_param("error_message", error_message[:250])  # Truncate long messages
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
        except Exception as e:
            logger.error(f"Failed to log error to MLflow: {str(e)}")