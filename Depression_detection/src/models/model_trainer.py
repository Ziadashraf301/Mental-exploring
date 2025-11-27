"""
DAG 4: Model Training Pipeline
Trains Classical ML and/or BERT models
Runs weekly on Sunday at 11:00 PM (after DAG 3)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import os

from Depression_detection.src.logger.train_logger import setup_train_logger
from Depression_detection.src.config.train_config_loader import reload_train_config
from Depression_detection.src.models.model_trainer import (
    train_multinomial_nb,
    train_sgd_classifier,
    run_cv_sklearn,
    train_bert_model
)
from Depression_detection.src.models.model_evaluator import get_report

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def train_classical_ml_task():
    """
    Train classical ML models (Multinomial NB, SGD Classifier).
    """
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("DAG 4: CLASSICAL ML TRAINING — START")

    # MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # Load vectorized datasets
    processed_data_path = CONFIG.processed_data_path
    models_path = CONFIG.models_dir

    LOGGER.info("Loading vectorized datasets...")

    with open(f"{processed_data_path}/X_train_vec.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(f"{processed_data_path}/X_test_vec.pkl", "rb") as f:
        X_test = pickle.load(f)

    with open(f"{processed_data_path}/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    with open(f"{processed_data_path}/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    LOGGER.info("✓ Vectorized data loaded")

    # Determine which models to train
    models_to_train = []

    if CONFIG.get_pipeline("classical_ml")["enabled"]:
        if CONFIG.get_model_params("multinomial_nb")["enabled"]:
            models_to_train.append(("multinomial_nb", train_multinomial_nb))
        
        if CONFIG.get_model_params("sgd_classifier")["enabled"]:
            models_to_train.append(("sgd_classifier", train_sgd_classifier))

    LOGGER.info(f"Models enabled for training: {[m for m, _ in models_to_train]}")

    # Train each model
    for model_name, train_fn in models_to_train:

        with mlflow.start_run(run_name=f"classical_{model_name}"):

            LOGGER.info(f"Training model: {model_name}")

            # Train
            model = train_fn(X_train, y_train)

            # Log parameters
            mlflow.log_params(CONFIG.get_model_params(model_name))

            # Cross-Validation
            if CONFIG.cv_params.get("enabled", False):
                LOGGER.info("Cross-Validation enabled — running cross-validation.")
                scoring = CONFIG.cv_params.get("scoring", "accuracy")
                cv_scores = run_cv_sklearn(model, X_train, y_train, scoring=scoring)
                if cv_scores:
                    mlflow.log_metric(f"cv_mean_{scoring}", float(np.mean(cv_scores)))
                    mlflow.log_metric(f"cv_std_{scoring}", float(np.std(cv_scores)))

            # Create plots directory
            if not os.path.exists(f"{CONFIG.plots_dir}/{model_name}/"):
                os.makedirs(f"{CONFIG.plots_dir}/{model_name}/")

            # Evaluation
            model_results = get_report(
                model=model,
                x_train=X_train,
                y_train=y_train,
                x_test=X_test,
                y_test=y_test,
                save_path=CONFIG.plots_dir + f"/{model_name}/{model_name}"
            )

            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": model_results["train"]["accuracy"],
                "train_f1": model_results["train"]["f1"],
                "train_precision": model_results["train"]["precision"],
                "train_recall": model_results["train"]["recall"],
                "train_auc": model_results["train"]["roc_auc"],
                "train_log_loss": model_results["train"]["logloss"],
                "test_accuracy": model_results["test"]["accuracy"],
                "test_f1": model_results["test"]["f1"],
                "test_precision": model_results["test"]["precision"],
                "test_recall": model_results["test"]["recall"],
                "test_auc": model_results["test"]["roc_auc"],
                "test_log_loss": model_results["test"]["logloss"],
            })

            # Save Artifacts - Plots
            plot_dir = CONFIG.plots_dir + f"/{model_name}/"
            for plot_file in os.listdir(plot_dir):
                mlflow.log_artifact(os.path.join(plot_dir, plot_file))
            
            LOGGER.info(f"✓ Completed training for {model_name}")

            # Save model
            if CONFIG.model_saving.get("save_models", True):
                model_path = f"{models_path}/{model_name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            
                mlflow.sklearn.log_model(
                    model,
                    f'Depression_Detection_{model_name}_Model',
                    registered_model_name=f'Depression_Detection_{model_name}_Model',
                    input_example=X_train[:5],
                    signature=infer_signature(X_train[:5], model.predict(X_train[:5])),
                )
                LOGGER.info(f"✓ Model saved at {model_path}")

    LOGGER.info("DAG 4: CLASSICAL ML TRAINING — COMPLETE")


def train_bert_task():
    """
    Train BERT model with LoRA.
    """
    CONFIG = reload_train_config("Depression_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("DAG 4: BERT TRAINING — START")

    # Check if BERT is enabled
    if not CONFIG.get_pipeline("bert")["enabled"]:
        LOGGER.info("BERT pipeline is disabled in config. Skipping BERT training.")
        return

    # MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # Load BERT datasets
    processed_data_path = CONFIG.processed_data_path
    models_path = CONFIG.models_dir

    LOGGER.info("Loading BERT datasets...")

    with open(f"{processed_data_path}/bert_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open(f"{processed_data_path}/bert_val_dataset.pkl", "rb") as f:
        val_dataset = pickle.load(f)

    with open(f"{processed_data_path}/bert_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    LOGGER.info("✓ BERT datasets loaded")

    # Create output directory
    output_dir = f"{models_path}/bert_depression_model"
    os.makedirs(output_dir, exist_ok=True)

    # Train BERT
    with mlflow.start_run(run_name="bert_model"):

        # Log BERT config parameters
        bert_cfg = CONFIG.pipelines["bert"]
        mlflow.log_params({
            "model_name": bert_cfg["tokenization"]["model_name"],
            "max_length": bert_cfg["tokenization"]["max_length"],
            "num_epochs": bert_cfg["training"]["num_epochs"],
            "batch_size": bert_cfg["training"]["batch_size"],
            "learning_rate": bert_cfg["training"]["learning_rate"],
            "lora_enabled": bert_cfg["lora"]["enabled"],
            "lora_r": bert_cfg["lora"]["r"] if bert_cfg["lora"]["enabled"] else None,
            "lora_alpha": bert_cfg["lora"]["lora_alpha"] if bert_cfg["lora"]["enabled"] else None,
        })

        # Train
        model, trainer = train_bert_model(train_dataset, val_dataset, output_dir)

        # Evaluate on test set
        LOGGER.info("Evaluating BERT on test set...")
        test_results = trainer.evaluate(test_dataset)

        LOGGER.info("Test Results:")
        for key, value in test_results.items():
            LOGGER.info(f"{key}: {value:.4f}")
            mlflow.log_metric(key, value)

        # Generate predictions for additional metrics
        predictions = trainer.predict(test_dataset)
        
        # Save model
        if CONFIG.model_saving.get("save_models", True):
            LOGGER.info("Saving BERT model...")
            
            # Load tokenizer to save with model
            from transformers import AutoTokenizer
            tokenizer_path_load = f"{models_path}/bert_tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_load)
            
            # Merge LoRA weights if enabled
            if bert_cfg["lora"]["enabled"]:
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
                LOGGER.info(f"✓ Merged BERT model saved at {output_dir}")
            else:
                model.save_pretrained(output_dir)
                LOGGER.info(f"✓ BERT model saved at {output_dir}")
            
            # Save tokenizer alongside model
            tokenizer.save_pretrained(output_dir)
            LOGGER.info(f"✓ Tokenizer saved with model at {output_dir}")
            
            # Log complete model + tokenizer to MLflow
            mlflow.pytorch.log_model(
                model,
                "bert_model",
                registered_model_name="Depression_Detection_BERT_Model"
            )
            
            # Log tokenizer artifacts
            mlflow.log_artifacts(output_dir, artifact_path="bert_model_with_tokenizer")
            LOGGER.info("✓ Model and tokenizer logged to MLflow")

        LOGGER.info("DAG 4: BERT TRAINING — COMPLETE")


# DAG Definition
with DAG(
    'depression_detection_04_training',
    default_args=default_args,
    description='Train Classical ML and BERT models',
    schedule='0 23 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['depression_detection', 'training'],
) as dag:

    # Task for Classical ML training
    train_classical = PythonOperator(
        task_id='train_classical_ml',
        python_callable=train_classical_ml_task,
    )

    # Task for BERT training
    train_bert = PythonOperator(
        task_id='train_bert',
        python_callable=train_bert_task,
    )

    # Run both training tasks in parallel
    [train_classical, train_bert]