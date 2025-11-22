"""
DAG 1: Data Loading Pipeline
Loads Teets
Runs: Weekly Sunday 11 PM
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pickle
import os

from Depression_detection.src.logger.train_logger import setup_train_logger
from Depression_detection.src.config.train_config_loader import reload_train_config
from Depression_detection.src.text.data_loader import load_data


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def load_data_task():
    """Load tweets data and save for next DAG."""
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("DAG 1: DATA LOADING - STARTING")
    
    # Load training data
    LOGGER.info("Loading training data...")
    
    train_data, test_data = load_data(
    raw_data_path=CONFIG.raw_data_path,
    target_column_names=CONFIG.data_loader['target_column_names'],
    text_column_names=CONFIG.data_loader['text_column_names'],
    labels_map=CONFIG.data_loader['labels_map'],
    test_size=CONFIG.data_loader['test_size'],      
    random_state=CONFIG.random_seeds['numpy_seed']
)

    # Save data for next DAG
    LOGGER.info("Saving processed data for next DAG...")

    processed_data_path = CONFIG.processed_data_path
    os.makedirs(processed_data_path, exist_ok=True)
    
    with open(f'{processed_data_path}/train_tweets.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{processed_data_path}/test_tweets.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    # Also save as CSV
    train_data.to_csv(f'{processed_data_path}/train_tweets.csv', index=False)
    test_data.to_csv(f'{processed_data_path}/test_tweets.csv', index=False)
    
    LOGGER.info(f"✓ Data saved to {processed_data_path}")
    LOGGER.info("DAG 1: DATA LOADING - COMPLETE")


with DAG(
    'depression_detection_01_data_loading',
    default_args=default_args,
    description='Load text data',
    schedule='0 22 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['depression_detection', 'data_loading'],
) as dag:
    
    load_text_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )

    trigger_preprocessing = TriggerDagRunOperator(
        task_id='trigger_preprocessing',
        trigger_dag_id='depression_detection_02_preprocessing',
    )

    load_text_data >> trigger_preprocessing