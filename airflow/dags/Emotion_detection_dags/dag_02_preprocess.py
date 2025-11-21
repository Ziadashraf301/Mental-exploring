"""
DAG 2: Data Preprocessing Pipeline
Prepares data for sklearn and tensorflow models
Runs: Weekly Sunday 10:15 PM (after DAG 1)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pickle
import os

from Emotion_detection.src.logger.train_logger import setup_train_logger
from Emotion_detection.src.config.train_config_loader import reload_train_config
from Emotion_detection.src.config.train_config_random_seed import set_random_seeds
from Emotion_detection.src.config.train_config_gpu import configure_gpu
from Emotion_detection.src.images.data_preprocessor import prepare_data_for_sklearn, prepare_data_for_tensorflow


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def preprocess_data_task():
    """Preprocess data for models"""
    CONFIG = reload_train_config("Emotion_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("DAG 2: DATA PREPROCESSING - STARTING")
    
    # Set random seeds
    set_random_seeds(CONFIG)
    configure_gpu(CONFIG)
    
    # Load data from DAG 1
    data_path = CONFIG.processed_data_path
    
    with open(f'{data_path}/train_images.pkl', 'rb') as f:
        train_images = pickle.load(f)
    with open(f'{data_path}/train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open(f'{data_path}/test_images.pkl', 'rb') as f:
        test_images = pickle.load(f)
    with open(f'{data_path}/test_labels.pkl', 'rb') as f:
        test_labels = pickle.load(f)
    
    LOGGER.info("✓ Data loaded from previous DAG")
    
    # Prepare sklearn data
    LOGGER.info("Preprocessing data for sklearn models...")
    X_train_sk, y_train_sk = prepare_data_for_sklearn(train_images, train_labels)
    X_test_sk, y_test_sk = prepare_data_for_sklearn(test_images, test_labels)
    LOGGER.info(f"✓ Sklearn data prepared: {X_train_sk.shape}")
    
    # Prepare tensorflow data
    LOGGER.info("Preprocessing data for tensorflow models...")
    X_train_tf, y_train_tf = prepare_data_for_tensorflow(train_images, train_labels)
    X_test_tf, y_test_tf = prepare_data_for_tensorflow(test_images, test_labels)
    LOGGER.info(f"✓ TensorFlow data prepared: {X_train_tf.shape}")
    
    # Save preprocessed data
    with open(f'{data_path}/X_train_sk.pkl', 'wb') as f:
        pickle.dump(X_train_sk, f)
    with open(f'{data_path}/y_train_sk.pkl', 'wb') as f:
        pickle.dump(y_train_sk, f)
    with open(f'{data_path}/X_test_sk.pkl', 'wb') as f:
        pickle.dump(X_test_sk, f)
    with open(f'{data_path}/y_test_sk.pkl', 'wb') as f:
        pickle.dump(y_test_sk, f)
    
    with open(f'{data_path}/X_train_tf.pkl', 'wb') as f:
        pickle.dump(X_train_tf, f)
    with open(f'{data_path}/y_train_tf.pkl', 'wb') as f:
        pickle.dump(y_train_tf, f)
    with open(f'{data_path}/X_test_tf.pkl', 'wb') as f:
        pickle.dump(X_test_tf, f)
    with open(f'{data_path}/y_test_tf.pkl', 'wb') as f:
        pickle.dump(y_test_tf, f)
    
    LOGGER.info(f"✓ Preprocessed data saved to {data_path}")
    LOGGER.info("DAG 2: DATA PREPROCESSING - COMPLETE")


with DAG(
    'emotion_detection_02_preprocessing',
    default_args=default_args,
    description='Preprocess data for training',
    schedule='15 22 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'preprocessing'],
) as dag:
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )
    
    trigger_train_sklearn_models = TriggerDagRunOperator(
        task_id='trigger_train_sklearn_models',
        trigger_dag_id='emotion_detection_03_train_sklearn',
    )

    trigger_train_cnn_model = TriggerDagRunOperator(
        task_id='trigger_train_cnn_model',
        trigger_dag_id='emotion_detection_04_train_cnn',
    )

    preprocess_data >> [ trigger_train_sklearn_models,trigger_train_cnn_model]
