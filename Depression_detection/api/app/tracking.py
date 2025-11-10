import wandb
from datetime import datetime
import os

class ModelMonitor:
    def __init__(self, project_name: str):
        # Initialize W&B
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        self.run = wandb.init(
            project=project_name,
            name=f"api-inference",
            job_type="inference",
            reinit=True
        )
    
    def log_prediction(
        self,
        user_id: str,
        text_length: int,
        prediction: int,
        confidence: float,
        inference_time: float
    ):
        """Log prediction to W&B"""
        wandb.log({
            "user_id": user_id,
            "text_length": text_length,
            "prediction": prediction,
            "confidence": confidence,
            "inference_time": inference_time,
            "timestamp": datetime.now().timestamp()
        })
    
    def log_feedback(self, prediction_id: str, correct_label: str):
        """Log user feedback"""
        wandb.log({
            "feedback": {
                "prediction_id": prediction_id,
                "correct_label": correct_label
            }
        })
    
    def log_error(self, error_type: str, error_message: str):
        """Log errors"""
        wandb.log({
            "error": {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def close(self):
        """Close W&B run"""
        self.run.finish()