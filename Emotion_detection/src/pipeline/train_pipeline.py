"""
Emotion Detection Pipeline - Main Entry Point
Reads all CONFIGuration from CONFIG.yaml
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import os

from src.logger.train_logger import setup_train_logger
from src.config.train_config_loader import reload_train_config


def run_pipeline(CONFIG_path: str = "config/train_config.yaml"):
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
    
    # ================== Load CONFIGuration ==================
    CONFIG = reload_train_config(CONFIG_path)
    
    # Setup logger
    LOGGER = setup_train_logger(
        log_file=CONFIG.log_file,
        log_level=CONFIG.log_level,
        console_output=CONFIG.console_output
    )

    LOGGER.info("=" * 70)
    LOGGER.info("EMOTION DETECTION PIPELINE - STARTING")
    LOGGER.info("=" * 70)
    LOGGER.info(f"CONFIGuration loaded from: {CONFIG_path}\n")
    
    # Print key settings
    LOGGER.info("Key Settings:")
    LOGGER.info(f"  - Experiment: {CONFIG.mlflow_experiment_name}")
    LOGGER.info(f"  - Logistic Regression: {'Enabled' if CONFIG.lr_enabled else 'Disabled'}")
    LOGGER.info(f"  - Feedforward NN: {'Enabled' if CONFIG.ffn_enabled else 'Disabled'}")
    LOGGER.info(f"  - CNN: {'Enabled' if CONFIG.cnn_enabled else 'Disabled'}")
    LOGGER.info(f"  - Data Augmentation: {'Enabled' if CONFIG.augmentation_enabled else 'Disabled'}")
    LOGGER.info(f"  - Cross Validation: {'Enabled' if CONFIG.cv_enabled else 'Disabled'}")
    
    from src.config.train_config_random_seed import set_random_seeds
    from src.config.train_config_gpu import configure_gpu

    # Set random seeds
    set_random_seeds(CONFIG)
    
    # CONFIGure GPU
    configure_gpu(CONFIG)
    
    # ================== Setup MLflow ==================
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name, )
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()

    # ================== Load Data ==================
    LOGGER.info("\n Loading Data...")
    LOGGER.info("-" * 70)

    from src.images.image_loader import load_images, load_labels
    from src.images.data_preprocessor import prepare_data_for_sklearn, prepare_data_for_tensorflow
    
    train_images, _ = load_images(CONFIG.train_images_path)
    train_labels = load_labels(CONFIG.train_labels_path)
    LOGGER.info(f"✓ Training: {len(train_images)} images")
    
    test_images, _ = load_images(CONFIG.test_images_path)
    test_labels = load_labels(CONFIG.test_labels_path)
    LOGGER.info(f"✓ Test: {len(test_images)} images")
    
    # ================== Preprocess Data ==================
    LOGGER.info("\n Preprocessing Data...")
    LOGGER.info("-" * 70)
    
    X_train_sk, y_train_sk = prepare_data_for_sklearn(train_images, train_labels)
    X_test_sk, y_test_sk = prepare_data_for_sklearn(test_images, test_labels)
    LOGGER.info(f"✓ Sklearn data prepared: {X_train_sk.shape}")
    
    X_train_tf, y_train_tf = prepare_data_for_tensorflow(train_images, train_labels)
    X_test_tf, y_test_tf = prepare_data_for_tensorflow(test_images, test_labels)
    LOGGER.info(f"✓ TensorFlow data prepared: {X_train_tf.shape}")
    
    # Store results
    results = {}
    

    from src.models.model_trainer import train_logistic_regression, train_ffn, build_cnn, train_cnn
    from src.models.model_evaluator import get_report
    
    # ================== Train Sklearn Models ==================
    if CONFIG.lr_enabled or CONFIG.ffn_enabled:
        LOGGER.info("\n Training Sklearn Models...")
        LOGGER.info("=" * 70)
        
        # Train Logistic Regression
        if CONFIG.lr_enabled:            
            with mlflow.start_run(run_name="logistic_regression"):
                LOGGER.info("\n  → Training Logistic Regression...")
                
                # Log hyperparameters
                mlflow.log_params({
                    "model_type": "logistic_regression",
                    "train_samples": len(X_train_sk),
                    "test_samples": len(X_test_sk),
                    "input_features": X_train_sk.shape[1],
                    "num_classes": len(np.unique(y_train_sk)),
                    **{f"lr_{k}": v for k, v in CONFIG.lr_params.items()}
                })
                
                # Train model
                lr_model = train_logistic_regression(X_train_sk, y_train_sk)
                
                # Evaluate
                metrics_lr = get_report(
                    lr_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                    model_type='sklearn',
                    save_path=f'{CONFIG.plots_dir}/lr_report'
                )
                
                # Log metrics
                mlflow.log_metrics({
                    "train_accuracy": metrics_lr["train"]["accuracy"],
                    "train_f1": metrics_lr["train"]["f1"],
                    "train_precision": metrics_lr["train"]["precision"],
                    "train_recall": metrics_lr["train"]["recall"],
                    "train_logloss": metrics_lr["train"]["logloss"],
                    "test_accuracy": metrics_lr["test"]["accuracy"],
                    "test_f1": metrics_lr["test"]["f1"],
                    "test_precision": metrics_lr["test"]["precision"],
                    "test_recall": metrics_lr["test"]["recall"],
                    "test_logloss": metrics_lr["test"]["logloss"],
                })
                
                # Log artifacts
                if os.path.exists(f"{CONFIG.plots_dir}/lr_report_metrics.png"):
                    mlflow.log_artifact(f"{CONFIG.plots_dir}/lr_report_metrics.png")
                
                # Save and register model
                if CONFIG.model_saving_params['save_sklearn_models']:
                    model_info = mlflow.sklearn.log_model(
                        lr_model, 
                        "LogisticRegression_EmotionDetection",
                        registered_model_name="LogisticRegression_EmotionDetection"
                    )
                    LOGGER.info(f"  ✓ Model registered: {model_info.model_uri}")
                
                results['lr_metrics'] = metrics_lr
                LOGGER.info("  ✓ Logistic Regression complete")
        
        # Train Feedforward NN
        if CONFIG.ffn_enabled:
            with mlflow.start_run(run_name="feedforward_neural_network"):
                LOGGER.info("\n  → Training Feedforward Neural Network...")
                
                # Log hyperparameters
                mlflow.log_params({
                    "model_type": "feedforward_nn",
                    "train_samples": len(X_train_sk),
                    "test_samples": len(X_test_sk),
                    "input_features": X_train_sk.shape[1],
                    "num_classes": len(np.unique(y_train_sk)),
                    **{f"ffn_{k}": v for k, v in CONFIG.ffn_params.items()}
                })
                
                # Train model
                ffn_model = train_ffn(X_train_sk, y_train_sk)
                
                # Evaluate
                metrics_ffn = get_report(
                    ffn_model, X_train_sk, y_train_sk, X_test_sk, y_test_sk,
                    model_type='sklearn',
                    save_path=f'{CONFIG.plots_dir}/ffn_report'
                )
                
                # Log metrics
                mlflow.log_metrics({
                    "train_accuracy": metrics_ffn["train"]["accuracy"],
                    "train_f1": metrics_ffn["train"]["f1"],
                    "train_precision": metrics_ffn["train"]["precision"],
                    "train_recall": metrics_ffn["train"]["recall"],
                    "train_logloss": metrics_ffn["train"]["logloss"],
                    "test_accuracy": metrics_ffn["test"]["accuracy"],
                    "test_f1": metrics_ffn["test"]["f1"],
                    "test_precision": metrics_ffn["test"]["precision"],
                    "test_recall": metrics_ffn["test"]["recall"],
                    "test_logloss": metrics_ffn["test"]["logloss"],
                })
                
                # Log artifacts
                if os.path.exists(f"{CONFIG.plots_dir}/ffn_report_metrics.png"):
                    mlflow.log_artifact(f"{CONFIG.plots_dir}/ffn_report_metrics.png")
                
                # Save and register model
                if CONFIG.model_saving_params['save_sklearn_models']:
                    model_info = mlflow.sklearn.log_model(
                        ffn_model, 
                        "FeedforwardNN_EmotionDetection",
                        registered_model_name="FeedforwardNN_EmotionDetection"
                    )
                    LOGGER.info(f"  ✓ Model registered: {model_info.model_uri}")
                
                results['ffn_metrics'] = metrics_ffn
                LOGGER.info("  ✓ Feedforward NN complete")
    
    # ================== Train CNN ==================
    if CONFIG.cnn_enabled:
        LOGGER.info("\n Training CNN Model...")
        LOGGER.info("=" * 70)
        
        with mlflow.start_run(run_name="cnn_model"):
            # Log hyperparameters
            cnn_params = {
                "model_type": "cnn",
                "train_samples": len(X_train_tf),
                "test_samples": len(X_test_tf),
                "image_height": X_train_tf.shape[1],
                "image_width": X_train_tf.shape[2],
                "image_channels": X_train_tf.shape[3],
                "num_classes": len(np.unique(y_train_tf)),
            }
            
            # Add training parameters (flatten nested dicts)
            for key, value in CONFIG.cnn_training_params.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        cnn_params[f"cnn_{key}_{nested_key}"] = str(nested_value)
                else:
                    cnn_params[f"cnn_{key}"] = value
            
            mlflow.log_params(cnn_params)
            
            # Build and train
            cnn_model = build_cnn()
            
            if CONFIG.cv_enabled:
                from sklearn.model_selection import KFold

                k = CONFIG.cv_params["k_folds"]
                shuffle = CONFIG.cv_params["shuffle"]
                random_state = CONFIG.cv_params["random_state"]

                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

                fold_train_acc = []
                fold_val_acc = []

                LOGGER.info(f"Running {k}-Fold Cross Validation...\n")

                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tf)):
                    
                    LOGGER.info(f"🔸 Fold {fold+1}/{k}")

                    X_train_fold, X_val_fold = X_train_tf[train_idx], X_train_tf[val_idx]
                    y_train_fold, y_val_fold = y_train_tf[train_idx], y_train_tf[val_idx]

                    # Child MLflow run for each fold
                    with mlflow.start_run(run_name=f"cnn_fold_{fold+1}", nested=True):

                        # Train CNN for this fold
                        model_fold, history = train_cnn(
                            model=build_cnn(),                        # fresh model per fold
                            X_train=X_train_fold,
                            y_train=y_train_fold,
                            X_val=X_val_fold,
                            y_val=y_val_fold,
                            batch_size=CONFIG.cnn_batch_size,
                            epochs=CONFIG.cnn_epochs,
                            augmentation_params=CONFIG.augmentation_params
                        )

                        # Evaluate fold
                        metrics_cnn = get_report(
                            model_fold,
                            X_train_fold, y_train_fold,
                            X_val_fold, y_val_fold,
                            model_type="tensorflow",
                            save_path=f"{CONFIG.plots_dir}/cnn_fold{fold+1}_report.png"
                        )

                        tr_acc = metrics_cnn["train"]["accuracy"]
                        va_acc = metrics_cnn["test"]["accuracy"]

                        fold_train_acc.append(tr_acc)
                        fold_val_acc.append(va_acc)

                        # Log metrics
                        mlflow.log_metrics({
                            "train_accuracy": tr_acc,
                            "val_accuracy": va_acc
                        })

                        # Log plot
                        plot_path = f"{CONFIG.plots_dir}/cnn_fold{fold+1}_report.png"
                        if os.path.exists(plot_path):
                            mlflow.log_artifact(plot_path)

                        LOGGER.info(
                            f"Fold {fold+1} completed — Train Acc: {tr_acc:.4f}, Val Acc: {va_acc:.4f}"
                        )

                # Compute averages
                avg_train = float(np.mean(fold_train_acc))
                avg_val = float(np.mean(fold_val_acc))

                mlflow.log_metrics({
                    "avg_train_accuracy": avg_train,
                    "avg_val_accuracy": avg_val
                })

                LOGGER.info(
                    f"\n🏁 K-Fold Complete — Avg Train Acc: {avg_train:.4f}, "
                    f"Avg Val Acc: {avg_val:.4f}"
                )

                results["cnn_metrics"] = {
                    "train": {"accuracy": avg_train},
                    "test": {"accuracy": avg_val},
                }

            # -----------------------------------------------
            #       CASE 2 — NORMAL TRAINING (NO CV)
            # -----------------------------------------------
            else:
                LOGGER.info("Training CNN WITHOUT Cross Validation...")

                cnn_model, history = train_cnn(
                    model=cnn_model,
                    X_train=X_train_tf,
                    y_train=y_train_tf,
                    X_val=X_test_tf,
                    y_val=y_test_tf,
                    batch_size=CONFIG.cnn_batch_size,
                    epochs=CONFIG.cnn_epochs,
                    augmentation_params=CONFIG.augmentation_params
                )

                # Evaluate
                metrics_cnn = get_report(
                    cnn_model,
                    X_train_tf, y_train_tf,
                    X_test_tf, y_test_tf,
                    model_type="tensorflow",
                    save_path=f"{CONFIG.plots_dir}/cnn_report.png"
                )

                
                # Log metrics
                mlflow.log_metrics({
                    "train_accuracy": metrics_cnn["train"]["accuracy"],
                    "train_f1": metrics_cnn["train"]["f1"],
                    "train_precision": metrics_cnn["train"]["precision"],
                    "train_recall": metrics_cnn["train"]["recall"],
                    "train_logloss": metrics_cnn["train"]["logloss"],
                    "test_accuracy": metrics_cnn["test"]["accuracy"],
                    "test_f1": metrics_cnn["test"]["f1"],
                    "test_precision": metrics_cnn["test"]["precision"],
                    "test_recall": metrics_cnn["test"]["recall"],
                    "test_logloss": metrics_cnn["test"]["logloss"],
                })
                
                # Log artifacts
                if os.path.exists(f"{CONFIG.plots_dir}/cnn_report_metrics.png"):
                    mlflow.log_artifact(f"{CONFIG.plots_dir}/cnn_report_metrics.png")
                
                # Save model locally
                if CONFIG.model_saving_params['save_cnn_model']:
                    save_format = CONFIG.model_saving_params['save_format']
                    model_path = f"{CONFIG.models_dir}/CNN_emotion_detection.{save_format}"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(CONFIG.models_dir, exist_ok=True)
                    
                    cnn_model.save(model_path)
                    LOGGER.info(f"  ✓ Model saved locally to {model_path}")
                
                # Register model in MLflow
                if CONFIG.model_saving_params['save_cnn_model']:
                    # Create input example for signature
                    input_example = X_train_tf[:1]
                    
                    model_info = mlflow.tensorflow.log_model(
                        cnn_model,
                        "CNN_EmotionDetection",
                        registered_model_name="CNN_EmotionDetection",
                        input_example=input_example
                    )
                    LOGGER.info(f"  ✓ Model registered: {model_info.model_uri}")
                
            results['cnn_metrics'] = metrics_cnn
            LOGGER.info("  ✓ CNN training complete")
    
    # ================== Summary ==================
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
    
    # ================== Completion ==================
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
    
    return results