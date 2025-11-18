"""
DAG 5: Training Summary
Generates final report and logs best model
Runs: Weekly Sunday 11 PM (after DAG 4)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pickle
import os

from Emotion_detection.src.logger.train_logger import setup_train_logger
from Emotion_detection.src.config.train_config_loader import reload_train_config


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def generate_summary_task():
    """Generate training summary report"""
    CONFIG = reload_train_config("Emotion_detection/train_config.yaml")
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 5: TRAINING SUMMARY - STARTING")
    LOGGER.info("=" * 70)
    
    data_path = '/opt/airflow/data/Emotion_detection/processed'
    results = {}
    
    # Load sklearn results
    if os.path.exists(f'{data_path}/sklearn_results.pkl'):
        with open(f'{data_path}/sklearn_results.pkl', 'rb') as f:
            sklearn_results = pickle.load(f)
            results.update(sklearn_results)
        LOGGER.info("✓ Sklearn results loaded")
    
    # Load CNN results
    if os.path.exists(f'{data_path}/cnn_results.pkl'):
        with open(f'{data_path}/cnn_results.pkl', 'rb') as f:
            cnn_results = pickle.load(f)
            results.update(cnn_results)
        LOGGER.info("✓ CNN results loaded")
    
    # Generate summary
    LOGGER.info("\n Results Summary")
    LOGGER.info("=" * 70)
    
    if results:
        LOGGER.info(f"{'Model':<25} {'Test Accuracy':<20} {'Test Loss':<15}")
        LOGGER.info("-" * 70)
        
        best_model = ("None", 0.0)
        
        if 'lr_metrics' in results:
            acc = results['lr_metrics']['test']["accuracy"] * 100
            loss = results['lr_metrics']['test']['logloss']
            LOGGER.info(f"{'Logistic Regression':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['lr_metrics']['test']["accuracy"] > best_model[1]:
                best_model = ("Logistic Regression", results['lr_metrics']['test']["accuracy"])
        
        if 'ffn_metrics' in results:
            acc = results['ffn_metrics']['test']["accuracy"] * 100
            loss = results['ffn_metrics']['test']['logloss']
            LOGGER.info(f"{'Feedforward NN':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['ffn_metrics']['test']["accuracy"] > best_model[1]:
                best_model = ("Feedforward NN", results['ffn_metrics']['test']["accuracy"])
        
        if 'cnn_metrics' in results:
            acc = results['cnn_metrics']['test']["accuracy"] * 100
            loss = results['cnn_metrics']['test']['logloss']
            LOGGER.info(f"{'CNN':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['cnn_metrics']['test']["accuracy"] > best_model[1]:
                best_model = ("CNN", results['cnn_metrics']['test']["accuracy"])
        
        LOGGER.info("=" * 70)
        LOGGER.info(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]*100:.2f}% test accuracy")
        results['best_model'] = best_model
    
    LOGGER.info("\n Pipeline Complete!")
    LOGGER.info("=" * 70)
    LOGGER.info("✓ PIPELINE FINISHED SUCCESSFULLY")
    LOGGER.info("=" * 70)
    LOGGER.info(f"\n📊 View MLflow UI:")
    LOGGER.info(f"   mlflow ui --backend-store-uri {CONFIG.mlflow_tracking_uri}")
    LOGGER.info(f"   Then open: http://localhost:5000")
    LOGGER.info(f"\n📁 Output Locations:")
    LOGGER.info(f"   Models: {CONFIG.models_dir}/")
    LOGGER.info(f"   Plots:  {CONFIG.plots_dir}/")
    LOGGER.info(f"   Logs:   {CONFIG.log_file}")
    LOGGER.info(f"\n📦 Registered Models in MLflow Model Registry:")
    if CONFIG.lr_enabled and CONFIG.model_saving_params['save_sklearn_models']:
        LOGGER.info("   - LogisticRegression_EmotionDetection")
    if CONFIG.ffn_enabled and CONFIG.model_saving_params['save_sklearn_models']:
        LOGGER.info("   - FeedforwardNN_EmotionDetection")
    if CONFIG.cnn_enabled and CONFIG.model_saving_params['save_cnn_model']:
        LOGGER.info("   - CNN_EmotionDetection")
    LOGGER.info("=" * 70)
    
    # Save final results
    with open(f'{data_path}/final_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    LOGGER.info("=" * 70)
    LOGGER.info("DAG 5: TRAINING SUMMARY - COMPLETE")
    LOGGER.info("=" * 70)


with DAG(
    'emotion_detection_05_summary',
    default_args=default_args,
    description='Generate training summary and report',
    schedule='0 23 * * 0',  # Sunday 11 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['emotion_detection', 'summary'],
) as dag:
    
    wait_for_cnn = ExternalTaskSensor(
        task_id='wait_for_cnn',
        external_dag_id='emotion_detection_04_train_cnn',
        external_task_id='train_cnn_model',
        timeout=3600,
        mode='reschedule',
    )
    
    generate_summary = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_summary_task,
    )
    
    wait_for_cnn >> generate_summary
