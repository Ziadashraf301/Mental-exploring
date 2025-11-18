"""
DAG 4: CNN Model Training
Trains CNN model with optional cross-validation
Runs: Weekly Sunday 11 PM (after DAG 3)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pickle
import mlflow
import mlflow.tensorflow
import numpy as np
import os

from Emotion_detection.src.logger.train_logger import setup_train_logger
from Emotion_detection.src.config.train_config_loader import reload_train_config
from Emotion_detection.src.models.model_trainer import build_cnn, train_cnn
from Emotion_detection.src.models.model_evaluator import get_report


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def train_cnn_model_task():
    """Train CNN model"""
    CONFIG = reload_train_config("Emotion_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 4: CNN TRAINING - STARTING")
    LOGGER.info("=" * 70)
    
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()
    
    # Load preprocessed data
    data_path = '/opt/airflow/data/Emotion_detection/processed'
    
    with open(f'{data_path}/X_train_tf.pkl', 'rb') as f:
        X_train_tf = pickle.load(f)
    with open(f'{data_path}/y_train_tf.pkl', 'rb') as f:
        y_train_tf = pickle.load(f)
    with open(f'{data_path}/X_test_tf.pkl', 'rb') as f:
        X_test_tf = pickle.load(f)
    with open(f'{data_path}/y_test_tf.pkl', 'rb') as f:
        y_test_tf = pickle.load(f)
    
    LOGGER.info("✓ Preprocessed data loaded")
    
    results = {}
    
    if CONFIG.cnn_enabled:
        with mlflow.start_run(run_name="cnn_model"):
            
            cnn_params = {
                "model_type": "cnn",
                "train_samples": len(X_train_tf),
                "test_samples": len(X_test_tf),
                "image_height": X_train_tf.shape[1],
                "image_width": X_train_tf.shape[2],
                "image_channels": X_train_tf.shape[3],
                "num_classes": len(np.unique(y_train_tf)),
            }
            
            for key, value in CONFIG.cnn_training_params.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        cnn_params[f"cnn_{key}_{nested_key}"] = str(nested_value)
                else:
                    cnn_params[f"cnn_{key}"] = value
            
            mlflow.log_params(cnn_params)
            
            cnn_model = build_cnn()
            
            if CONFIG.cv_enabled:
                from sklearn.model_selection import KFold
                
                k = CONFIG.cv_params["k_folds"]
                shuffle = CONFIG.cv_params["shuffle"]
                random_state = CONFIG.cv_params["random_state"]
                
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                
                fold_train_acc = []
                fold_val_acc = []
                
                LOGGER.info(f"Running {k}-Fold Cross Validation...")
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tf)):
                    LOGGER.info(f"🔸 Fold {fold+1}/{k}")
                    
                    X_train_fold, X_val_fold = X_train_tf[train_idx], X_train_tf[val_idx]
                    y_train_fold, y_val_fold = y_train_tf[train_idx], y_train_tf[val_idx]
                    
                    with mlflow.start_run(run_name=f"cnn_fold_{fold+1}", nested=True):
                        model_fold, history = train_cnn(
                            model=build_cnn(),
                            X_train=X_train_fold,
                            y_train=y_train_fold,
                            X_val=X_val_fold,
                            y_val=y_val_fold,
                            batch_size=CONFIG.cnn_batch_size,
                            epochs=CONFIG.cnn_epochs,
                            augmentation_params=CONFIG.augmentation_params
                        )
                        
                        metrics_cnn = get_report(
                            model_fold,
                            X_train_fold, y_train_fold,
                            X_val_fold, y_val_fold,
                            model_type="tensorflow",
                            save_path=f"{CONFIG.plots_dir}/cnn_fold{fold+1}_report.png"
                        )
                        
                        tr_acc = metrics_cnn["train"]["accuracy"]
                        va_acc = metrics_cnn["test"]["accuracy"]
                        
                        fold_train_acc.append(tr_acc)
                        fold_val_acc.append(va_acc)
                        
                        mlflow.log_metrics({
                            "train_accuracy": tr_acc,
                            "val_accuracy": va_acc
                        })
                        
                        plot_path = f"{CONFIG.plots_dir}/cnn_fold{fold+1}_report.png"
                        if os.path.exists(plot_path):
                            mlflow.log_artifact(plot_path)
                        
                        LOGGER.info(f"Fold {fold+1} completed — Train Acc: {tr_acc:.4f}, Val Acc: {va_acc:.4f}")
                
                avg_train = float(np.mean(fold_train_acc))
                avg_val = float(np.mean(fold_val_acc))
                
                mlflow.log_metrics({
                    "avg_train_accuracy": avg_train,
                    "avg_val_accuracy": avg_val
                })
                
                LOGGER.info(f"🏁 K-Fold Complete — Avg Train Acc: {avg_train:.4f}, Avg Val Acc: {avg_val:.4f}")
                
                results["cnn_metrics"] = {
                    "train": {"accuracy": avg_train},
                    "test": {"accuracy": avg_val},
                }
            
            else:
                LOGGER.info("Training CNN WITHOUT Cross Validation...")
                
                cnn_model, history = train_cnn(
                    model=cnn_model,
                    X_train=X_train_tf,
                    y_train=y_train_tf,
                    X_val=X_test_tf,
                    y_val=y_test_tf,
                    batch_size=CONFIG.cnn_batch_size,
                    epochs=CONFIG.cnn_epochs,
                    augmentation_params=CONFIG.augmentation_params
                )
                
                metrics_cnn = get_report(
                    cnn_model,
                    X_train_tf, y_train_tf,
                    X_test_tf, y_test_tf,
                    model_type="tensorflow",
                    save_path=f"{CONFIG.plots_dir}/cnn_report_metrics.png"
                )
                
                mlflow.log_metrics({
                    "train_accuracy": metrics_cnn["train"]["accuracy"],
                    "train_f1": metrics_cnn["train"]["f1"],
                    "train_precision": metrics_cnn["train"]["precision"],
                    "train_recall": metrics_cnn["train"]["recall"],
                    "train_logloss": metrics_cnn["train"]["logloss"],
                    "test_accuracy": metrics_cnn["test"]["accuracy"],
                    "test_f1": metrics_cnn["test"]["f1"],
                    "test_precision": metrics_cnn["test"]["precision"],
                    "test_recall": metrics_cnn["test"]["recall"],
                    "test_logloss": metrics_cnn["test"]["logloss"],
                })
                
                if os.path.exists(f"{CONFIG.plots_dir}/cnn_report_metrics.png"):
                    mlflow.log_artifact(f"{CONFIG.plots_dir}/cnn_report_metrics.png")
                
                if CONFIG.model_saving_params['save_cnn_model']:
                    save_format = CONFIG.model_saving_params['save_format']
                    model_path = f"{CONFIG.models_dir}/CNN_EMOTION_DETECTION.{save_format}"
                    
                    os.makedirs(CONFIG.models_dir, exist_ok=True)
                    
                    cnn_model.save(model_path)
                    LOGGER.info(f"✓ Model saved locally to {model_path}")
                
                if CONFIG.model_saving_params['save_cnn_model']:
                    input_example = X_train_tf[:1]
                    
                    model_info = mlflow.tensorflow.log_model(
                        cnn_model,
                        "CNN_EmotionDetection",
                        registered_model_name="CNN_EmotionDetection",
                        input_example=input_example
                    )
                    LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
                
                results['cnn_metrics'] = metrics_cnn
                LOGGER.info("✓ CNN training complete")
    
    # Save results
    with open(f'{data_path}/cnn_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 4: CNN TRAINING - COMPLETE")
    LOGGER.info("=" * 70)


with DAG(
    'emotion_detection_04_train_cnn',
    default_args=default_args,
    description='Train CNN model',
    schedule='0 23 * * 0',  # Sunday 11 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'cnn_training'],
) as dag:
    
    wait_for_sklearn = ExternalTaskSensor(
        task_id='wait_for_sklearn',
        external_dag_id='emotion_detection_03_train_sklearn',
        external_task_id='train_sklearn_models',
        timeout=3600,
        mode='reschedule',
    )
    
    train_cnn_model = PythonOperator(
        task_id='train_cnn_model',
        python_callable=train_cnn_model_task,
    )
    
    wait_for_sklearn >> train_cnn_model
