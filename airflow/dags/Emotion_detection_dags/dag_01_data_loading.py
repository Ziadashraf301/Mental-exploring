"""
DAG 1: Data Loading Pipeline
Loads training and test images/labels
Runs: Weekly Sunday 11 PM
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pickle
import os

from Emotion_detection.src.logger.train_logger import setup_train_logger
from Emotion_detection.src.config.train_config_loader import reload_train_config
from Emotion_detection.src.images.image_loader import load_images, load_labels


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def load_data_task():
    """Load images and labels"""
    CONFIG = reload_train_config("Emotion_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 1: DATA LOADING - STARTING")
    LOGGER.info("=" * 70)
    
    # Load training data
    LOGGER.info("Loading training data...")
    train_images, _ = load_images(CONFIG.train_images_path)
    train_labels = load_labels(CONFIG.train_labels_path)
    LOGGER.info(f"✓ Training: {len(train_images)} images")
    
    # Load test data
    LOGGER.info("Loading test data...")
    test_images, _ = load_images(CONFIG.test_images_path)
    test_labels = load_labels(CONFIG.test_labels_path)
    LOGGER.info(f"✓ Test: {len(test_images)} images")
    
    # Save data for next DAG
    data_path = '/opt/airflow/data/Emotion_detection/processed'
    os.makedirs(data_path, exist_ok=True)
    
    with open(f'{data_path}/train_images.pkl', 'wb') as f:
        pickle.dump(train_images, f)
    with open(f'{data_path}/train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)
    with open(f'{data_path}/test_images.pkl', 'wb') as f:
        pickle.dump(test_images, f)
    with open(f'{data_path}/test_labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f)
    
    LOGGER.info(f"✓ Data saved to {data_path}")
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 1: DATA LOADING - COMPLETE")
    LOGGER.info("=" * 70)


with DAG(
    'emotion_detection_01_data_loading',
    default_args=default_args,
    description='Load training and test data',
    schedule='0 23 * * 0',  # Sunday 11 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'data_loading'],
) as dag:
    
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )