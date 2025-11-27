"""
DAG 3: Vectorization Pipeline
Transforms preprocessed tweets using TF-IDF (Classical ML) and/or Tokenization (BERT)
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

from Depression_detection.src.logger.train_logger import setup_train_logger
from Depression_detection.src.config.train_config_loader import reload_train_config
from Depression_detection.src.models.model_vectorizer import (
    build_tfidf_vectorizer,
    transform_data,
    build_bert_tokenizer,
    create_bert_datasets
)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def vectorize_classical_ml_task():
    """
    Vectorize data for classical ML models using TF-IDF.
    """
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    # If disabled → log & exit cleanly
    if not CONFIG.get_pipeline("classical_ml")["enabled"]:
        LOGGER.info("DAG 3: TF-IDF Vectorization and classical ml disabled — skipping.")
        return

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

        # Register vectorizer in MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="TFIDF_Vectorizer",
            registered_model_name="TFIDF_Vectorizer_Depression"
        )

        LOGGER.info(f"✓ Vectorizer registered: {model_info.model_uri}")
        LOGGER.info("DAG 3: TF-IDF VECTORIZATION — COMPLETE")


def prepare_bert_datasets_task():
    """
    Prepare tokenized datasets for BERT training.
    """
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    # If disabled → log & exit cleanly
    if not CONFIG.get_pipeline("bert")["enabled"]:
        LOGGER.info("DAG 4: BERT dataset preparation disabled — skipping.")
        return  
    
    LOGGER.info("DAG 3: BERT TOKENIZATION — START")

    processed_data_path = CONFIG.processed_data_path
    tokenizer_path = CONFIG.models_dir

    # MLflow Setup
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name="bert_tokenization"):

        LOGGER.info("Loading preprocessed tweets...")

        with open(f"{processed_data_path}/X_train.pkl", "rb") as f:
            X_train = pickle.load(f)

        with open(f"{processed_data_path}/X_test.pkl", "rb") as f:
            X_test = pickle.load(f)

        with open(f"{processed_data_path}/y_train.pkl", "rb") as f:
            y_train = pickle.load(f)

        with open(f"{processed_data_path}/y_test.pkl", "rb") as f:
            y_test = pickle.load(f)

        # Split test into validation and test (50/50)
        from sklearn.model_selection import train_test_split
        X_val, X_test_final, y_val, y_test_final = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        LOGGER.info("✓ Preprocessed data loaded and split into train/val/test")

        # Build tokenizer
        LOGGER.info("Loading BERT tokenizer...")
        tokenizer = build_bert_tokenizer()

        # Log tokenizer parameters
        bert_cfg = CONFIG.pipelines["bert"]["tokenization"]
        mlflow.log_params({
            "bert_model_name": bert_cfg["model_name"],
            "bert_max_length": bert_cfg["max_length"],
        })

        # Create datasets
        LOGGER.info("Creating BERT datasets...")
        train_dataset, val_dataset, test_dataset = create_bert_datasets(
            X_train, y_train,
            X_val, y_val,
            X_test_final, y_test_final,
            tokenizer
        )

        LOGGER.info("✓ BERT datasets created")

        # Log dataset sizes
        mlflow.log_metrics({
            "bert_train_samples": len(train_dataset),
            "bert_val_samples": len(val_dataset),
            "bert_test_samples": len(test_dataset)
        })

        # Save datasets
        LOGGER.info("Saving BERT datasets...")

        with open(f"{processed_data_path}/bert_train_dataset.pkl", "wb") as f:
            pickle.dump(train_dataset, f)

        with open(f"{processed_data_path}/bert_val_dataset.pkl", "wb") as f:
            pickle.dump(val_dataset, f)

        with open(f"{processed_data_path}/bert_test_dataset.pkl", "wb") as f:
            pickle.dump(test_dataset, f)

        # Save tokenizer
        tokenizer_save_path = f"{tokenizer_path}/bert_tokenizer"
        tokenizer.save_pretrained(tokenizer_save_path)
        LOGGER.info(f"✓ Tokenizer saved at {tokenizer_save_path}")

        # Log tokenizer to MLflow
        mlflow.log_artifacts(tokenizer_save_path, artifact_path="bert_tokenizer")
        LOGGER.info("✓ Tokenizer logged to MLflow")

        # Log vocab size as metric
        vocab_size = len(tokenizer.get_vocab())
        mlflow.log_metric("bert_vocab_size", vocab_size)
        LOGGER.info(f"✓ Vocabulary size: {vocab_size}")

        LOGGER.info("DAG 3: BERT TOKENIZATION — COMPLETE")


# DAG Definition
with DAG(
    'depression_detection_03_vectorization',
    default_args=default_args,
    description='Apply TF-IDF vectorization and BERT tokenization',
    schedule='40 22 * * 0', 
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['depression_detection', 'vectorization', 'tokenization'],
) as dag:

    # Task for Classical ML
    vectorize_classical = PythonOperator(
        task_id='vectorize_classical_ml',
        python_callable=vectorize_classical_ml_task,
    )

    # Task for BERT
    prepare_bert = PythonOperator(
        task_id='prepare_bert_datasets',
        python_callable=prepare_bert_datasets_task,
    )

    # Trigger training DAG
    trigger_train = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='depression_detection_04_training',
    )

    # Run both vectorization tasks in parallel, then trigger training
    [vectorize_classical, prepare_bert] >> trigger_train