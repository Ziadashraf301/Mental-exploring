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

from Sentiment_analysis.src.logger.train_logger import setup_train_logger
from Sentiment_analysis.src.config.train_config_loader import reload_train_config
from Sentiment_analysis.src.tweets.data_loader import load_data


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
    CONFIG = reload_train_config("Sentiment_analysis/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("DAG 1: DATA LOADING - STARTING")
    
    # Load training data
    LOGGER.info("Loading training data...")
    
    train_tweets, test_tweets = load_data(
        data_path = CONFIG.raw_data_path,
        dataset_columns=CONFIG.data_loader['dataset_columns'],
        dataset_encoding=CONFIG.data_loader['dataset_encoding'],
        engine=CONFIG.data_loader['engine'],
        sentiment_col=CONFIG.data_loader['sentiment_col'],
        text_col=CONFIG.data_loader['text_col'],
        test_size=CONFIG.data_loader['test_size'],      
        random_state=CONFIG.random_seeds['numpy_seed']
        )

    # Save data for next DAG
    LOGGER.info("Saving processed data for next DAG...")

    processed_data_path = CONFIG.processed_data_path
    os.makedirs(processed_data_path, exist_ok=True)
    
    with open(f'{processed_data_path}/train_tweets.pkl', 'wb') as f:
        pickle.dump(train_tweets, f)
    with open(f'{processed_data_path}/test_tweets.pkl', 'wb') as f:
        pickle.dump(test_tweets, f)

    LOGGER.info(f"✓ Data saved to {processed_data_path}")
    LOGGER.info("DAG 1: DATA LOADING - COMPLETE")


with DAG(
    'sentiment_analysis_01_data_loading',
    default_args=default_args,
    description='Load tweets data',
    schedule='0 22 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sentiment_analysis', 'data_loading'],
) as dag:
    
    load_tweet_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )

    trigger_preprocessing = TriggerDagRunOperator(
        task_id='trigger_preprocessing',
        trigger_dag_id='sentiment_analysis_02_preprocessing',
    )

    load_tweet_data >> trigger_preprocessing