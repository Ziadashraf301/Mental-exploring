"""
DAG 2: Data Preprocessing Pipeline
Prepares tweets text for models
Runs weekly on Sunday at 22:10 PM (after DAG 1)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import time
import pickle
from tqdm import tqdm 

from Depression_detection.src.logger.train_logger import setup_train_logger
from Depression_detection.src.config.train_config_loader import reload_train_config
from Depression_detection.src.config.train_config_random_seed import set_random_seeds
from Depression_detection.src.text.data_preprocessor import clean_tweets


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
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("DAG 2: DATA PREPROCESSING — START")

    # Set random seed for reproducibility
    set_random_seeds(CONFIG)

    # Load data produced by DAG 1
    processed_data_path = CONFIG.processed_data_path

    LOGGER.info("Loading raw tweets from previous DAG...")

    with open(f"{processed_data_path}/train_tweets.pkl", "rb") as f:
        train_tweets = pickle.load(f)

    with open(f"{processed_data_path}/test_tweets.pkl", "rb") as f:
        test_tweets = pickle.load(f)

    LOGGER.info("✓ Data loaded successfully")

    # PROCESS TRAIN TWEETS
    LOGGER.info("Preprocessing TRAIN tweets...")

    start = time.time()

    processed_train_text = []
    for tweet in tqdm(train_tweets["filtered_tweet"]):
        processed = clean_tweets(tweet)
        processed_train_text.append(processed)

    y_train = list(train_tweets["is_depression"])

    LOGGER.info(f"✓ Train preprocessing done in {round(time.time() - start, 2)} seconds")

    # PROCESS TEST TWEETS
    LOGGER.info("Preprocessing TEST tweets...")

    processed_test_text = []
    for tweet in tqdm(test_tweets["text"]):
        processed = clean_tweets(tweet)
        processed_test_text.append(processed)

    y_test = list(test_tweets["is_depression"])

    LOGGER.info("✓ Test preprocessing complete")

    # SAVE OUTPUTS
    LOGGER.info("Saving preprocessed data...")

    # Save Scikit-style preprocessed datasets
    with open(f"{processed_data_path}/X_train.pkl", "wb") as f:
        pickle.dump(processed_train_text, f)

    with open(f"{processed_data_path}/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    with open(f"{processed_data_path}/X_test.pkl", "wb") as f:
        pickle.dump(processed_test_text, f)

    with open(f"{processed_data_path}/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    LOGGER.info("✓ Saved preprocessed text data")
    LOGGER.info("✓ All preprocessed datasets saved successfully")
    LOGGER.info("DAG 2: DATA PREPROCESSING — COMPLETE")


with DAG(
    'depression_detection_02_preprocessing',
    default_args=default_args,
    description='Preprocess data for training',
    schedule='10 22 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['depression_detection', 'preprocessing'],
) as dag:
    
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )

    trigger_vectorization = TriggerDagRunOperator(
        task_id='trigger_vectorization',
        trigger_dag_id='depression_detection_03_vectorization',
    )

    preprocess_data >> trigger_vectorization