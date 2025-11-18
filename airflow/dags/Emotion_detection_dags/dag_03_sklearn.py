"""
DAG 3: Sklearn Models Training
Trains Logistic Regression and Feedforward NN
Runs: Weekly Sunday 11 PM (after DAG 2)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pickle
import mlflow
import mlflow.sklearn
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
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 3: SKLEARN TRAINING - STARTING")
    LOGGER.info("=" * 70)
    
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # Load preprocessed data
    data_path = '/opt/airflow/data/Emotion_detection/processed'
    
    with open(f'{data_path}/X_train_sk.pkl', 'rb') as f:
        X_train_sk = pickle.load(f)
    with open(f'{data_path}/y_train_sk.pkl', 'rb') as f:
        y_train_sk = pickle.load(f)
    with open(f'{data_path}/X_test_sk.pkl', 'rb') as f:
        X_test_sk = pickle.load(f)
    with open(f'{data_path}/y_test_sk.pkl', 'rb') as f:
        y_test_sk = pickle.load(f)
    
    LOGGER.info("✓ Preprocessed data loaded")
    
    results = {}
    
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
            
            metrics_lr = get_report(
                lr_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                model_type='sklearn',
                save_path=f'{CONFIG.plots_dir}/lr_report'
            )
            
            mlflow.log_metrics({
                "train_accuracy": metrics_lr["train"]["accuracy"],
                "train_f1": metrics_lr["train"]["f1"],
                "train_precision": metrics_lr["train"]["precision"],
                "train_recall": metrics_lr["train"]["recall"],
                "train_logloss": metrics_lr["train"]["logloss"],
                "test_accuracy": metrics_lr["test"]["accuracy"],
                "test_f1": metrics_lr["test"]["f1"],
                "test_precision": metrics_lr["test"]["precision"],
                "test_recall": metrics_lr["test"]["recall"],
                "test_logloss": metrics_lr["test"]["logloss"],
            })
            
            if os.path.exists(f"{CONFIG.plots_dir}/lr_report_metrics.png"):
                mlflow.log_artifact(f"{CONFIG.plots_dir}/lr_report_metrics.png")
            
            if CONFIG.model_saving_params['save_sklearn_models']:
                model_info = mlflow.sklearn.log_model(
                    lr_model, 
                    "LogisticRegression_EmotionDetection",
                    registered_model_name="LogisticRegression_EmotionDetection"
                )
                LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
            
            results['lr_metrics'] = metrics_lr
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
            
            metrics_ffn = get_report(
                ffn_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                model_type='sklearn',
                save_path=f'{CONFIG.plots_dir}/ffn_report'
            )
            
            mlflow.log_metrics({
                "train_accuracy": metrics_ffn["train"]["accuracy"],
                "train_f1": metrics_ffn["train"]["f1"],
                "train_precision": metrics_ffn["train"]["precision"],
                "train_recall": metrics_ffn["train"]["recall"],
                "train_logloss": metrics_ffn["train"]["logloss"],
                "test_accuracy": metrics_ffn["test"]["accuracy"],
                "test_f1": metrics_ffn["test"]["f1"],
                "test_precision": metrics_ffn["test"]["precision"],
                "test_recall": metrics_ffn["test"]["recall"],
                "test_logloss": metrics_ffn["test"]["logloss"],
            })
            
            if os.path.exists(f"{CONFIG.plots_dir}/ffn_report_metrics.png"):
                mlflow.log_artifact(f"{CONFIG.plots_dir}/ffn_report_metrics.png")
            
            if CONFIG.model_saving_params['save_sklearn_models']:
                model_info = mlflow.sklearn.log_model(
                    ffn_model, 
                    "FeedforwardNN_EmotionDetection",
                    registered_model_name="FeedforwardNN_EmotionDetection"
                )
                LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
            
            results['ffn_metrics'] = metrics_ffn
            LOGGER.info("✓ Feedforward NN complete")
    
    # Save results
    with open(f'{data_path}/sklearn_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 3: SKLEARN TRAINING - COMPLETE")
    LOGGER.info("=" * 70)


with DAG(
    'emotion_detection_03_train_sklearn',
    default_args=default_args,
    description='Train sklearn models',
    schedule='0 23 * * 0',  # Sunday 11 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'sklearn_training'],
) as dag:
    
    wait_for_preprocessing = ExternalTaskSensor(
        task_id='wait_for_preprocessing',
        external_dag_id='emotion_detection_02_preprocessing',
        external_task_id='preprocess_data',
        timeout=3600,
        mode='reschedule',
    )
    
    train_sklearn_models = PythonOperator(
        task_id='train_sklearn_models',
        python_callable=train_sklearn_models_task,
    )
    
    wait_for_preprocessing >> train_sklearn_models
