"""
DAG 4: Model Training Pipeline
Trains models on vectorized tweets
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

from Sentiment_analysis.src.logger.train_logger import setup_train_logger
from Sentiment_analysis.src.config.train_config_loader import reload_train_config
from Sentiment_analysis.src.models.model_trainer import train_multinomial_nb, train_linear_svc, train_logistic_regression, run_cv
from Sentiment_analysis.src.models.model_evaluator import get_report

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def train_data_task():

    # Load config and logger
    CONFIG = reload_train_config("Sentiment_analysis/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("DAG 4: TRAINING PIPELINE — START")

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

    # TRAIN ENABLED MODELS
    models_to_train = []

    if CONFIG.mnb_enabled:
        models_to_train.append(("MultinomialNB", train_multinomial_nb))

    if CONFIG.svc_enabled:
        models_to_train.append(("LinearSVC", train_linear_svc))

    if CONFIG.lr_enabled:
        models_to_train.append(("LogisticRegression", train_logistic_regression))

    LOGGER.info(f"Models enabled for training: {[m for m, _ in models_to_train]}")

    # TRAIN EACH MODEL
    for model_name, train_fn in models_to_train:

        with mlflow.start_run(run_name=model_name):

            LOGGER.info(f"Training model: {model_name}")

            # Train
            model = train_fn(X_train, y_train)

            # Log parameters
            mlflow.log_params(CONFIG.get_params_for(model_name))

            # Cross-Validation
            if CONFIG.cv_params.get("enabled", False):
                LOGGER.info("Cross-Validation enabled — running cross-validation.")
                scoring = CONFIG.cv_params.get("scoring", "accuracy")
                cv_scores = run_cv(model, X_train, y_train, scoring=scoring)
                if cv_scores:
                    mlflow.log_metric(f"cv_mean_{scoring}", float(np.mean(cv_scores)))
                    mlflow.log_metric(f"cv_std_{scoring}", float(np.std(cv_scores)))

            if os.path.exists(f"{CONFIG.plots_dir}/{model_name}/"):
                LOGGER.info(f"Plots directory for {model_name} already exists.")
            else:
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
                        f'Sentiment_analysis{model_name}_Model',
                        registered_model_name=f'Sentiment_analysis{model_name}_Model',
                        input_example=X_train[:5],
                        signature=infer_signature(X_train[:5], model.predict(X_train[:5])),
                    )
                LOGGER.info(f"✓ Model saved at {model_path}")

    LOGGER.info("DAG 4: TRAINING COMPLETE")


with DAG(
    'sentiment_analysis_04_training',
    default_args=default_args,
    description='Train models on vectorized tweets',
    schedule='0 23 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sentiment_analysis', 'training'],
) as dag:

    train_data = PythonOperator(
        task_id='train_data',
        python_callable=train_data_task,
    )

    train_data