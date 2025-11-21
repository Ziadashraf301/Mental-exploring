"""
DAG 3: Sklearn Models Training
Trains Logistic Regression and Feedforward NN
Runs: Weekly Sunday 10:45 PM (after DAG 2)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import os

from Emotion_detection.src.logger.train_logger import setup_train_logger
from Emotion_detection.src.config.train_config_loader import reload_train_config
from Emotion_detection.src.models.model_trainer import train_logistic_regression, train_ffn
from Emotion_detection.src.models.model_evaluator import get_report


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def train_sklearn_models_task():
    """Train sklearn models"""
    CONFIG = reload_train_config("Emotion_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("DAG 3: SKLEARN TRAINING - STARTING")
    
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # Load preprocessed data
    data_path = CONFIG.processed_data_path
    
    with open(f'{data_path}/X_train_sk.pkl', 'rb') as f:
        X_train_sk = pickle.load(f)
    with open(f'{data_path}/y_train_sk.pkl', 'rb') as f:
        y_train_sk = pickle.load(f)
    with open(f'{data_path}/X_test_sk.pkl', 'rb') as f:
        X_test_sk = pickle.load(f)
    with open(f'{data_path}/y_test_sk.pkl', 'rb') as f:
        y_test_sk = pickle.load(f)
    
    LOGGER.info("✓ Preprocessed data loaded")
    
    # Train Logistic Regression
    if CONFIG.lr_enabled:
        with mlflow.start_run(run_name="logistic_regression"):
            LOGGER.info("→ Training Logistic Regression...")
            
            mlflow.log_params({
                "model_type": "logistic_regression",
                "train_samples": len(X_train_sk),
                "test_samples": len(X_test_sk),
                "input_features": X_train_sk.shape[1],
                "num_classes": len(np.unique(y_train_sk)),
                **{f"lr_{k}": v for k, v in CONFIG.lr_params.items()}
            })
            
            lr_model = train_logistic_regression(X_train_sk, y_train_sk)
            
            if os.path.exists(f"{CONFIG.plots_dir}/logistic_regression/"):
                LOGGER.info(f"Plots directory for logistic_regression already exists.")
            else:
                os.makedirs(f"{CONFIG.plots_dir}/logistic_regression/")

            metrics_lr = get_report(
                lr_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                model_type='sklearn',
                save_path=f'{CONFIG.plots_dir}/logistic_regression/logistic_regression'
            )
            
            mlflow.log_metrics({
                "train_accuracy": metrics_lr["train"]["accuracy"],
                "train_f1": metrics_lr["train"]["f1"],
                "train_precision": metrics_lr["train"]["precision"],
                "train_recall": metrics_lr["train"]["recall"],
                "train_logloss": metrics_lr["train"]["logloss"],
                "train_roc_auc": metrics_lr["train"]["roc_auc"],
                "test_accuracy": metrics_lr["test"]["accuracy"],
                "test_f1": metrics_lr["test"]["f1"],
                "test_precision": metrics_lr["test"]["precision"],
                "test_recall": metrics_lr["test"]["recall"],
                "test_logloss": metrics_lr["test"]["logloss"],
                "test_roc_auc": metrics_lr["test"]["roc_auc"],
            })

            # Save Artifacts - Plots
            plot_dir = CONFIG.plots_dir + "/logistic_regression/"
            for plot_file in os.listdir(plot_dir):
                mlflow.log_artifact(os.path.join(plot_dir, plot_file))

            if CONFIG.model_saving_params['save_sklearn_models']:
                model_info = mlflow.sklearn.log_model(
                    lr_model, 
                    "LogisticRegression_EmotionDetection",
                    registered_model_name="LogisticRegression_EmotionDetection",
                    input_example=X_train_sk[:5],
                    signature=infer_signature(X_train_sk[:5], lr_model.predict(X_train_sk[:5])),
                )
                LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
            LOGGER.info("✓ Logistic Regression complete")
    
    # Train Feedforward NN
    if CONFIG.ffn_enabled:
        with mlflow.start_run(run_name="feedforward_neural_network"):
            LOGGER.info("→ Training Feedforward Neural Network...")
            
            mlflow.log_params({
                "model_type": "feedforward_nn",
                "train_samples": len(X_train_sk),
                "test_samples": len(X_test_sk),
                "input_features": X_train_sk.shape[1],
                "num_classes": len(np.unique(y_train_sk)),
                **{f"ffn_{k}": v for k, v in CONFIG.ffn_params.items()}
            })
            
            ffn_model = train_ffn(X_train_sk, y_train_sk)
                     
            if os.path.exists(f"{CONFIG.plots_dir}/ffn/"):
                LOGGER.info(f"Plots directory for ffn already exists.")
            else:
                os.makedirs(f"{CONFIG.plots_dir}/ffn/")

            metrics_ffn = get_report(
                ffn_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                model_type='sklearn',
                save_path=f'{CONFIG.plots_dir}/ffn/ffn'
            )
            
            mlflow.log_metrics({
                "train_accuracy": metrics_ffn["train"]["accuracy"],
                "train_f1": metrics_ffn["train"]["f1"],
                "train_precision": metrics_ffn["train"]["precision"],
                "train_recall": metrics_ffn["train"]["recall"],
                "train_logloss": metrics_ffn["train"]["logloss"],
                "train_roc_auc": metrics_ffn["train"]["roc_auc"],
                "test_accuracy": metrics_ffn["test"]["accuracy"],
                "test_f1": metrics_ffn["test"]["f1"],
                "test_precision": metrics_ffn["test"]["precision"],
                "test_recall": metrics_ffn["test"]["recall"],
                "test_logloss": metrics_ffn["test"]["logloss"],
                "test_roc_auc": metrics_ffn["test"]["roc_auc"],
            })

            # Save Artifacts - Plots
            plot_dir = CONFIG.plots_dir + "/ffn/"
            for plot_file in os.listdir(plot_dir):
                mlflow.log_artifact(os.path.join(plot_dir, plot_file))
   
            if CONFIG.model_saving_params['save_sklearn_models']:
                model_info = mlflow.sklearn.log_model(
                    ffn_model, 
                    "FeedforwardNN_EmotionDetection",
                    registered_model_name="FeedforwardNN_EmotionDetection",
                    input_example=X_train_sk[:5],
                    signature=infer_signature(X_train_sk[:5], ffn_model.predict(X_train_sk[:5])),
                )
                LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
            
            LOGGER.info("✓ Feedforward NN complete")

    LOGGER.info("DAG 3: SKLEARN TRAINING - COMPLETE")


with DAG(
    'emotion_detection_03_train_sklearn',
    default_args=default_args,
    description='Train sklearn models',
    schedule='45 22 * * 0',  # Sunday 10:45 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'sklearn_training'],
) as dag:
    
  
    train_sklearn_models = PythonOperator(
        task_id='train_sklearn_models',
        python_callable=train_sklearn_models_task,
    )
    
    train_sklearn_models