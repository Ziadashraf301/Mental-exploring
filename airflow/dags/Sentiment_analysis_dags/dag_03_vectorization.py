"""
DAG 3: TF-IDF Vectorization Pipeline
Transforms preprocessed tweets using TF-IDF
Runs weekly on Sunday at 22:40 PM (after DAG 2)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pickle
import json
import mlflow
import mlflow.sklearn

from Sentiment_analysis.src.logger.train_logger import setup_train_logger
from Sentiment_analysis.src.config.train_config_loader import reload_train_config
from Sentiment_analysis.src.models.model_vectorizer import build_tfidf_vectorizer, transform_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def vectorize_data_task():

    CONFIG = reload_train_config("Sentiment_analysis/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("DAG 3: TF-IDF VECTORIZATION — START")

    processed_data_path = CONFIG.processed_data_path
    vectorizer_path = CONFIG.models_dir

    # MLflow Setup
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name="tfidf_vectorization"):

        LOGGER.info("Loading preprocessed tweets...")

        with open(f"{processed_data_path}/X_train.pkl", "rb") as f:
            X_train = pickle.load(f)

        with open(f"{processed_data_path}/X_test.pkl", "rb") as f:
            X_test = pickle.load(f)

        with open(f"{processed_data_path}/y_train.pkl", "rb") as f:
            y_train = pickle.load(f)

        with open(f"{processed_data_path}/y_test.pkl", "rb") as f:
            y_test = pickle.load(f)

        LOGGER.info("✓ Preprocessed data loaded successfully")

        # Build TF-IDF Pipeline
        LOGGER.info("Building TF-IDF vectorizer...")
        vectorizer = build_tfidf_vectorizer()

        # Log TF-IDF parameters
        mlflow.log_params({
            "tfidf_max_features": vectorizer.max_features,
            "tfidf_min_df": vectorizer.min_df,
            "tfidf_max_df": vectorizer.max_df,
            "tfidf_ngram_range": str(vectorizer.ngram_range),
            "tfidf_use_idf": vectorizer.use_idf,
            "tfidf_norm": vectorizer.norm,
            "tfidf_sublinear_tf": vectorizer.sublinear_tf,
        })

        # Fit + Transform
        X_train_vec, X_test_vec = transform_data(vectorizer, X_train, X_test)

        LOGGER.info("✓ TF-IDF transformation complete")

        # Log feature sizes
        mlflow.log_metrics({
            "train_vectorized_features": X_train_vec.shape[1],
            "test_vectorized_features": X_test_vec.shape[1],
            "vocabulary_size": len(vectorizer.vocabulary_)
        })

        # Save vectorized datasets
        LOGGER.info("Saving vectorized datasets...")

        with open(f"{processed_data_path}/X_train_vec.pkl", "wb") as f:
            pickle.dump(X_train_vec, f)

        with open(f"{processed_data_path}/X_test_vec.pkl", "wb") as f:
            pickle.dump(X_test_vec, f)

        # Save and log vectorizer
        vectorizer_file = f"{vectorizer_path}/tfidf_vectorizer.pkl"
        with open(vectorizer_file, "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact(vectorizer_file)

        # Save vocabulary as JSON artifact
        vocab_clean = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        vocab_json = f"{vectorizer_path}/tfidf_vocabulary.json"
        with open(vocab_json, "w") as f:
            json.dump(vocab_clean, f)

        # mlflow.log_artifact(vocab_json)

        # Register vectorizer in MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="TFIDF_Vectorizer",
            registered_model_name="TFIDF_Vectorizer_Sentiment"
        )

        LOGGER.info(f"✓ Vectorizer registered: {model_info.model_uri}")
        LOGGER.info("DAG 3: TF-IDF VECTORIZATION — COMPLETE")


# DAG Definition
with DAG(
    'sentiment_analysis_03_vectorization',
    default_args=default_args,
    description='Apply TF-IDF vectorization',
    schedule='40 22 * * 0', 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sentiment_analysis', 'tfidf', 'vectorization'],
) as dag:

    vectorize_data = PythonOperator(
        task_id='vectorize_data',
        python_callable=vectorize_data_task,
    )

    trigger_train = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='sentiment_analysis_04_training',
    )

    vectorize_data >> trigger_train