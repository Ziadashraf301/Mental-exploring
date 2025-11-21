"""
DAG 4: CNN Model Training
Trains CNN model with optional cross-validation
Runs: Weekly Sunday 10:45 PM (after DAG 2)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pickle
import mlflow
import mlflow.tensorflow
import numpy as np
import os
from sklearn.model_selection import KFold
from mlflow.models import infer_signature

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
    
    LOGGER.info("DAG 4: CNN TRAINING - STARTING")
    
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()
    
    # Load preprocessed data
    data_path = CONFIG.processed_data_path
    
    with open(f'{data_path}/X_train_tf.pkl', 'rb') as f:
        X_train_tf = pickle.load(f)
    with open(f'{data_path}/y_train_tf.pkl', 'rb') as f:
        y_train_tf = pickle.load(f)
    with open(f'{data_path}/X_test_tf.pkl', 'rb') as f:
        X_test_tf = pickle.load(f)
    with open(f'{data_path}/y_test_tf.pkl', 'rb') as f:
        y_test_tf = pickle.load(f)
    
    LOGGER.info("✓ Preprocessed data loaded")
    
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
                k = CONFIG.cv_params["k_folds"]
                shuffle = CONFIG.cv_params["shuffle"]
                random_state = CONFIG.cv_params["random_state"]
                
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                
                fold_train_acc = []
                fold_val_acc = []
                fold_train_f1 = []
                fold_val_f1 = []
                fold_train_logloss = []
                fold_val_logloss = []
                fold_train_precision = []
                fold_val_precision = []
                fold_train_recall = []
                fold_val_recall = []
                fold_train_auc = []
                fold_val_auc = []
                
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

                        if os.path.exists(f"{CONFIG.plots_dir}/cnn_cv/"):
                            LOGGER.info(f"Plots directory for cnn_cv already exists.")
                        else:
                            os.makedirs(f"{CONFIG.plots_dir}/cnn_cv/")
                        
                        metrics_cnn = get_report(
                            model_fold,
                            X_train_fold, y_train_fold,
                            X_val_fold, y_val_fold,
                            model_type="tensorflow",
                            save_path=f"{CONFIG.plots_dir}/cnn_cv/cnn_fold{fold+1}"
                        )
                        
                        tr_acc = metrics_cnn["train"]["accuracy"]
                        va_acc = metrics_cnn["test"]["accuracy"]
                        tr_f1 = metrics_cnn["train"]["f1"]
                        va_f1 = metrics_cnn["test"]["f1"]
                        tr_logloss = metrics_cnn["train"]["logloss"]
                        va_logloss = metrics_cnn["test"]["logloss"]
                        tr_precision = metrics_cnn["train"]["precision"]
                        va_precision = metrics_cnn["test"]["precision"]
                        tr_recall = metrics_cnn["train"]["recall"]
                        va_recall = metrics_cnn["test"]["recall"]
                        tr_auc = metrics_cnn["train"]["roc_auc"]
                        va_auc = metrics_cnn["test"]["roc_auc"]
                        
                        fold_train_acc.append(tr_acc)
                        fold_val_acc.append(va_acc)
                        fold_train_f1.append(tr_f1)
                        fold_val_f1.append(va_f1)
                        fold_train_logloss.append(tr_logloss)
                        fold_val_logloss.append(va_logloss)
                        fold_train_precision.append(tr_precision)
                        fold_val_precision.append(va_precision)
                        fold_train_recall.append(tr_recall)
                        fold_val_recall.append(va_recall)
                        fold_train_auc.append(tr_auc)
                        fold_val_auc.append(va_auc)
                        
                        mlflow.log_metrics({
                            "train_accuracy": tr_acc,
                            "val_accuracy": va_acc,
                            "train_f1": tr_f1,
                            "val_f1": va_f1,
                            "train_logloss": tr_logloss,
                            "val_logloss": va_logloss,
                            "train_precision": tr_precision,
                            "val_precision": va_precision,
                            "train_recall": tr_recall,
                            "val_recall": va_recall,
                            "train_roc_auc": tr_auc,
                            "val_roc_auc": va_auc,
                        })
                                    
                        # Save Artifacts - Plots
                        plot_dir = CONFIG.plots_dir + "/cnn_cv/"
                        for plot_file in os.listdir(plot_dir):
                            mlflow.log_artifact(os.path.join(plot_dir, plot_file))
                        
                        LOGGER.info(f"Fold {fold+1} completed — Train Acc: {tr_acc:.4f}, Val Acc: {va_acc:.4f}")
                
                avg_train = float(np.mean(fold_train_acc))
                avg_val = float(np.mean(fold_val_acc))
                avg_train_f1 = float(np.mean(fold_train_f1))
                avg_val_f1 = float(np.mean(fold_val_f1))
                avg_train_logloss = float(np.mean(fold_train_logloss))
                avg_val_logloss = float(np.mean(fold_val_logloss))
                avg_train_precision = float(np.mean(fold_train_precision))
                avg_val_precision = float(np.mean(fold_val_precision))
                avg_train_recall = float(np.mean(fold_train_recall))
                avg_val_recall = float(np.mean(fold_val_recall))
                avg_train_auc = float(np.mean(fold_train_auc))
                avg_val_auc = float(np.mean(fold_val_auc))
                
                mlflow.log_metrics({
                    "avg_train_accuracy": avg_train,
                    "avg_val_accuracy": avg_val,
                    "avg_train_f1": avg_train_f1,
                    "avg_val_f1": avg_val_f1,
                    "avg_train_logloss": avg_train_logloss,
                    "avg_val_logloss": avg_val_logloss,
                    "avg_train_precision": avg_train_precision,
                    "avg_val_precision": avg_val_precision,
                    "avg_train_recall": avg_train_recall,
                    "avg_val_recall": avg_val_recall,
                    "avg_train_roc_auc": avg_train_auc,
                    "avg_val_roc_auc": avg_val_auc,
                })
                
                LOGGER.info(f"🏁 K-Fold Complete — Avg Train Acc: {avg_train:.4f}, Avg Val Acc: {avg_val:.4f}")       
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
                
                if os.path.exists(f"{CONFIG.plots_dir}/cnn/"):
                    LOGGER.info(f"Plots directory for cnn already exists.")
                else:
                    os.makedirs(f"{CONFIG.plots_dir}/cnn/")

                metrics_cnn = get_report(
                    cnn_model,
                    X_train_tf, y_train_tf,
                    X_test_tf, y_test_tf,
                    model_type="tensorflow",
                    save_path=f"{CONFIG.plots_dir}/cnn/"
                )
                
                mlflow.log_metrics({
                    "train_accuracy": metrics_cnn["train"]["accuracy"],
                    "train_f1": metrics_cnn["train"]["f1"],
                    "train_precision": metrics_cnn["train"]["precision"],
                    "train_recall": metrics_cnn["train"]["recall"],
                    "train_logloss": metrics_cnn["train"]["logloss"],
                    "train_roc_auc": metrics_cnn["train"]["roc_auc"],
                    "test_accuracy": metrics_cnn["test"]["accuracy"],
                    "test_f1": metrics_cnn["test"]["f1"],
                    "test_precision": metrics_cnn["test"]["precision"],
                    "test_recall": metrics_cnn["test"]["recall"],
                    "test_logloss": metrics_cnn["test"]["logloss"],
                    "test_roc_auc": metrics_cnn["test"]["roc_auc"],
                })
                
                # Save Artifacts - Plots
                plot_dir = CONFIG.plots_dir + "/cnn/"
                for plot_file in os.listdir(plot_dir):
                    mlflow.log_artifact(os.path.join(plot_dir, plot_file))
                
                # Save Model
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
                        input_example=input_example,
                        signature=infer_signature(input_example, cnn_model.predict(input_example)),
                    )
                    LOGGER.info(f"✓ Model registered: {model_info.model_uri}")
                
    LOGGER.info("DAG 4: CNN TRAINING - COMPLETE")


with DAG(
    'emotion_detection_04_train_cnn',
    default_args=default_args,
    description='Train CNN model',
    schedule='45 22 * * 0',  # Sunday 10:45 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'cnn_training'],
) as dag:
  
    train_cnn_model = PythonOperator(
        task_id='train_cnn_model',
        python_callable=train_cnn_model_task,
    )
    
    train_cnn_model
