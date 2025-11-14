"""
Emotion Detection Pipeline - Main Entry Point
Reads all configuration from config.yaml
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from images.image_loader import load_images, load_labels
from images.data_preprocessor import prepare_data_for_sklearn, prepare_data_for_tensorflow
from models.model_trainer import train_logistic_regression, train_ffn, build_cnn, train_cnn
from models.model_evaluator import get_report
from utils.pipeline_logger import pipeline_logger
from utils.config_loader import get_config, reload_config


def set_random_seeds(config):
    """Set random seeds for reproducibility"""
    seeds = config.random_seeds
    
    # Python
    random.seed(seeds['python_seed'])
    
    # NumPy
    np.random.seed(seeds['numpy_seed'])
    
    # TensorFlow
    tf.random.set_seed(seeds['tensorflow_seed'])
    
    # Set for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seeds['python_seed'])
    
    pipeline_logger.info(f"Random seeds set: {seeds}")


def configure_gpu(config):
    """Configure GPU settings"""
    if config.use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory fraction
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(1024 * config.gpu_memory_fraction)
                    )]
                )
                pipeline_logger.info(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                pipeline_logger.warning(f"GPU configuration error: {e}")
        else:
            pipeline_logger.warning("No GPU found, using CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        pipeline_logger.info("GPU disabled, using CPU")


def run_pipeline(config_path: str = "config.yaml"):
    """
    Complete ML pipeline for emotion detection.
    All settings are loaded from config.yaml
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    dict
        Dictionary containing all results
    """
    
    # ================== 0. Load Configuration ==================
    config = reload_config(config_path)
    pipeline_logger.info("=" * 70)
    pipeline_logger.info("EMOTION DETECTION PIPELINE - STARTING")
    pipeline_logger.info("=" * 70)
    pipeline_logger.info(f"Configuration loaded from: {config_path}\n")
    
    # Print key settings
    pipeline_logger.info("Key Settings:")
    pipeline_logger.info(f"  - Experiment: {config.mlflow_experiment_name}")
    pipeline_logger.info(f"  - Logistic Regression: {'Enabled' if config.lr_enabled else 'Disabled'}")
    pipeline_logger.info(f"  - Feedforward NN: {'Enabled' if config.ffn_enabled else 'Disabled'}")
    pipeline_logger.info(f"  - CNN: {'Enabled' if config.cnn_enabled else 'Disabled'}")
    pipeline_logger.info(f"  - Data Augmentation: {'Enabled' if config.augmentation_enabled else 'Disabled'}")
    pipeline_logger.info(f"  - Cross Validation: {'Enabled' if config.cv_enabled else 'Disabled'}")
    
    # Set random seeds
    set_random_seeds(config)
    
    # Configure GPU
    configure_gpu(config)
    
    # ================== 1. Setup MLflow ==================
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    # ================== 2. Load Data ==================
    pipeline_logger.info("\n[STEP 1/6] Loading Data...")
    pipeline_logger.info("-" * 70)
    
    train_images, _ = load_images(config.train_images_path)
    train_labels = load_labels(config.train_labels_path)
    pipeline_logger.info(f"✓ Training: {len(train_images)} images")
    
    test_images, _ = load_images(config.test_images_path)
    test_labels = load_labels(config.test_labels_path)
    pipeline_logger.info(f"✓ Test: {len(test_images)} images")
    
    # ================== 3. Preprocess Data ==================
    pipeline_logger.info("\n[STEP 2/6] Preprocessing Data...")
    pipeline_logger.info("-" * 70)
    
    X_train_sk, y_train_sk = prepare_data_for_sklearn(train_images, train_labels)
    X_test_sk, y_test_sk = prepare_data_for_sklearn(test_images, test_labels)
    pipeline_logger.info(f"✓ Sklearn data prepared: {X_train_sk.shape}")
    
    X_train_tf, y_train_tf = prepare_data_for_tensorflow(train_images, train_labels)
    X_test_tf, y_test_tf = prepare_data_for_tensorflow(test_images, test_labels)
    pipeline_logger.info(f"✓ TensorFlow data prepared: {X_train_tf.shape}")
    
    # Store results
    results = {}
    
    # ================== 4. Train Sklearn Models (Separate Experiments) ==================
    if config.lr_enabled or config.ffn_enabled:
        pipeline_logger.info("\n[STEP 3/6] Training Sklearn Models...")
        pipeline_logger.info("=" * 70)
        
        # Train Logistic Regression in separate experiment
        if config.lr_enabled:            
            with mlflow.start_run(run_name="logistic_regression"):
                pipeline_logger.info("\n  → Training Logistic Regression...")
                
                # Log dataset info
                mlflow.log_params({
                    "train_samples": len(X_train_sk),
                    "test_samples": len(X_test_sk),
                    "image_shape": str(X_train_sk[0].shape),
                    "num_classes": len(np.unique(y_train_sk))
                })
                
                # Train model
                lr_model = train_logistic_regression(X_train_sk, y_train_sk)
                
                # Evaluate
                metrics_lr = get_report(
                    lr_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                    model_type='sklearn',
                    save_path=f'{config.plots_dir}/lr_report'
                )
                
                # Log to MLflow
                if config.model_saving_params['save_sklearn_models']:
                    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
                mlflow.log_params({f"lr_{k}": v for k, v in config.lr_params.items()})
                mlflow.log_metrics({
                    "test_accuracy": metrics_lr['acc_test'],
                    "test_loss": metrics_lr['loss_test'],
                    "train_accuracy": metrics_lr['acc_train'],
                    "train_loss": metrics_lr['loss_train']
                })
                mlflow.log_artifact(f"{config.plots_dir}/lr_report_metrics.png")
                
                results['lr_metrics'] = metrics_lr
                pipeline_logger.info("  ✓ Logistic Regression complete")
        
        # Train Feedforward NN in separate experiment
        if config.ffn_enabled:
            with mlflow.start_run(run_name="feedforward_neural_network"):
                pipeline_logger.info("\n  → Training Feedforward Neural Network...")
                
                # Log dataset info
                mlflow.log_params({
                    "train_samples": len(X_train_sk),
                    "test_samples": len(X_test_sk),
                    "image_shape": str(X_train_sk[0].shape),
                    "num_classes": len(np.unique(y_train_sk))
                })
                
                # Train model
                ffn_model = train_ffn(X_train_sk, y_train_sk)
                
                # Evaluate
                metrics_ffn = get_report(
                    ffn_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                    model_type='sklearn',
                    save_path=f'{config.plots_dir}/ffn_report'
                )
                
                # Log to MLflow
                if config.model_saving_params['save_sklearn_models']:
                    mlflow.sklearn.log_model(ffn_model, "ffn_model")
                mlflow.log_params({f"ffn_{k}": v for k, v in config.ffn_params.items()})
                mlflow.log_metrics({
                    "test_accuracy": metrics_ffn['acc_test'],
                    "test_loss": metrics_ffn['loss_test'],
                    "train_accuracy": metrics_ffn['acc_train'],
                    "train_loss": metrics_ffn['loss_train']
                })
                mlflow.log_artifact(f"{config.plots_dir}/ffn_report_metrics.png")
                
                results['ffn_metrics'] = metrics_ffn
                pipeline_logger.info("  ✓ Feedforward NN complete")
    
    # ================== 5. Train CNN ==================
    if config.cnn_enabled:
        pipeline_logger.info("\n[STEP 4/6] Training CNN Model...")
        pipeline_logger.info("=" * 70)
        
        with mlflow.start_run(run_name="cnn_model"):
            # Log dataset info
            mlflow.log_params({
                    "train_samples": len(X_train_tf),
                    "test_samples": len(X_test_tf),
                    "image_shape": str(X_train_tf[0].shape),
                    "num_classes": len(np.unique(y_train_tf)),
            })
            
            cnn_model = build_cnn()
            cnn_model, history = train_cnn(cnn_model, X_train_tf, y_train_tf)
            
            # Evaluate CNN
            metrics_cnn = get_report(
                cnn_model, X_train_tf, y_train_tf, X_test_tf, y_test_tf,
                model_type='tensorflow',
                save_path=f'{config.plots_dir}/cnn_report'
            )
            
            # Plot training history
            if hasattr(history, 'history') and config.cv_params['enabled']:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
                axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
                axes[0].set_title('CNN Training Accuracy', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Accuracy')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(history.history['loss'], label='Train', linewidth=2, marker='o')
                axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
                axes[1].set_title('CNN Training Loss', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = f'{config.plots_dir}/cnn_training_history.png'
                plt.savefig(plot_path, dpi=config.evaluation_params['plot_dpi'])
                plt.close()
                mlflow.log_artifact(plot_path)
            
            # Log to MLflow
            if config.model_saving_params['save_cnn_model']:
                mlflow.tensorflow.log_model(cnn_model, "cnn_model")
            
            mlflow.log_params({
                f"cnn_{k}": v for k, v in config.cnn_training_params.items()
                if not isinstance(v, dict)
            })

            cnn_metrics = { k: v for k, v in metrics_cnn.items()}

            mlflow.log_metrics({
                    "test_accuracy": cnn_metrics['acc_test'],
                    "test_loss": cnn_metrics['loss_test'],
                    "train_accuracy": cnn_metrics['acc_train'],
                    "train_loss": cnn_metrics['loss_train']
                })
            
            mlflow.log_artifact(f"{config.plots_dir}/cnn_report_metrics.png")
            
            # Save model locally
            if config.model_saving_params['save_cnn_model']:
                save_format = config.model_saving_params['save_format']
                model_path = f"{config.models_dir}/CNN_emotion_detection.{save_format}"
                cnn_model.save(model_path)
                mlflow.log_artifact(model_path)
                pipeline_logger.info(f"  ✓ Model saved to {model_path}")
            
            results['cnn_metrics'] = metrics_cnn
            pipeline_logger.info("  ✓ CNN training complete")
    
    # ================== 6. Summary ==================
    pipeline_logger.info("\n[STEP 5/6] Results Summary")
    pipeline_logger.info("=" * 70)
    
    if results:
        pipeline_logger.info(f"{'Model':<25} {'Test Accuracy':<20} {'Test Loss':<15}")
        pipeline_logger.info("-" * 70)
        
        best_model = ("None", 0.0)
        
        if 'lr_metrics' in results:
            acc = results['lr_metrics']['acc_test'] * 100
            loss = results['lr_metrics']['loss_test']
            pipeline_logger.info(f"{'Logistic Regression':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['lr_metrics']['acc_test'] > best_model[1]:
                best_model = ("Logistic Regression", results['lr_metrics']['acc_test'])
        
        if 'ffn_metrics' in results:
            acc = results['ffn_metrics']['acc_test'] * 100
            loss = results['ffn_metrics']['loss_test']
            pipeline_logger.info(f"{'Feedforward NN':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['ffn_metrics']['acc_test'] > best_model[1]:
                best_model = ("Feedforward NN", results['ffn_metrics']['acc_test'])
        
        if 'cnn_metrics' in results:
            acc = results['cnn_metrics']['acc_test'] * 100
            loss = results['cnn_metrics']['loss_test']
            pipeline_logger.info(f"{'CNN':<25} {acc:>6.2f}%{'':<13} {loss:>10.4f}")
            if results['cnn_metrics']['acc_test'] > best_model[1]:
                best_model = ("CNN", results['cnn_metrics']['acc_test'])
        
        pipeline_logger.info("=" * 70)
        pipeline_logger.info(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]*100:.2f}% test accuracy")
        results['best_model'] = best_model
    
    # ================== 7. Completion ==================
    pipeline_logger.info("\n[STEP 6/6] Pipeline Complete!")
    pipeline_logger.info("=" * 70)
    pipeline_logger.info("✓ PIPELINE FINISHED SUCCESSFULLY")
    pipeline_logger.info("=" * 70)
    pipeline_logger.info(f"\n📊 View MLflow UI:")
    pipeline_logger.info(f"   mlflow ui --backend-store-uri {config.mlflow_tracking_uri}")
    pipeline_logger.info(f"   Then open: http://localhost:5000")
    pipeline_logger.info(f"\n📁 Output Locations:")
    pipeline_logger.info(f"   Models: {config.models_dir}/")
    pipeline_logger.info(f"   Plots:  {config.plots_dir}/")
    pipeline_logger.info(f"   Logs:   {config.logs_dir}/pipeline.log")
    pipeline_logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    """
    Run the pipeline using settings from config.yaml
    To use a different config file: python run_pipeline.py path/to/config.yaml
    """
    import sys
    
    # Check if custom config path provided
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    print(f"\n🚀 Starting pipeline with configuration: {config_path}\n")
    
    # Run pipeline
    results = run_pipeline(config_path)
    
    # Print final results
    if 'best_model' in results:
        print(f"\n{'='*70}")
        print(f"🎯 Best Model: {results['best_model'][0]}")
        print(f"🎯 Test Accuracy: {results['best_model'][1]*100:.2f}%")
        print(f"{'='*70}\n")