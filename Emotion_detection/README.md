# Emotion Detection

Our emotion detection model analyzes facial expressions in images to identify emotional states. This technology has applications in identifying cyberbullying, enhancing mental health care, and facilitating better emotion management, especially for individuals with conditions like social anxiety or autism.

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/2ebc6909-3733-480a-8bac-d1d949d29366

## Model Performance Comparison

We developed and evaluated three machine learning models for emotion detection:

### 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| 🏆 CNN (Production) | 87.52% | 87.04% | 91.37% | 89.15% | 94.52% |
| Feedforward Neural Network | 71.79% | 70.70% | 84.98% | 77.18% | 79.31% |
| Logistic Regression | 70.26% | 72.01% | 76.93% | 74.39% | 75.66% |

### Model Selection

We selected **CNN** as our production model due to:

✅ **Best Performance**: 87.52% accuracy on test set  
✅ **Superior Metrics**: Highest precision (87.04%), recall (91.37%), and ROC AUC (94.52%)  
✅ **Spatial Feature Learning**: Automatically extracts meaningful patterns from images  
✅ **Robust to Variations**: Handles different lighting, angles, and facial features  
✅ **Deep Learning Power**: Captures complex emotional expressions effectively

### Confusion Matrix Analysis

Our CNN model shows strong performance:

```
                 Predicted
                Sad     Happy
Actual  Sad    36.22%   7.64%
        Happy   4.85%  51.30%
```

**Interpretation:**
- True Negatives: 36.22% (correctly identified sad expressions)
- False Positives: 7.64% (sad expressions misclassified as happy)
- False Negatives: 4.85% (happy expressions misclassified as sad)
- True Positives: 51.30% (correctly identified happy expressions)

## Pipeline Overview

**Complete Pipeline:**
1. [Data loading](#dag-1-data-loading-pipeline)
2. [Data preprocessing](#dag-2-data-preprocessing-pipeline)
3. [Model training - Sklearn](#dag-3-sklearn-models-training)
4. [Model training - CNN](#dag-4-cnn-model-training)

## Dataset

- **Image size**: 48×48 pixels
- **Color mode**: RGB (3 channels)
- **Classes**: Sad (0), Happy (1)
- **Train/Test Split**: 80%/20%

## Automated Training Pipeline (Apache Airflow)

We've implemented an automated training pipeline using Apache Airflow with 4 sequential DAGs running weekly on Sunday nights.

### Pipeline Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│              WEEKLY TRAINING PIPELINE                     │
│                 (Every Sunday Night)                      │
└──────────────────────────────────────────────────────────┘

DAG 1: Data Loading         → 10:00 PM Sunday
  ↓ (Trigger)
DAG 2: Preprocessing        → 10:15 PM Sunday
  ↓ (Trigger)              ↓ (Trigger)
DAG 3: Train Sklearn       DAG 4: Train CNN
       10:45 PM                   10:45 PM
```

### Default Configuration

```python
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}
```

---

## DAG 1: Data Loading Pipeline

**Schedule**: `0 22 * * 0` (10:00 PM every Sunday)  
**Purpose**: Load training and test images with labels

### Key Tasks

```python
def load_data_task():
    # Load training data
    train_images, _ = load_images(CONFIG.train_images_path)
    train_labels = load_labels(CONFIG.train_labels_path)
    
    # Load test data
    test_images, _ = load_images(CONFIG.test_images_path)
    test_labels = load_labels(CONFIG.test_labels_path)
    
    # Save for next DAG
    save_to_pickle(train_images, train_labels, test_images, test_labels)
```

**Output:**
- `train_images.pkl` → Training images
- `train_labels.pkl` → Training labels
- `test_images.pkl` → Test images
- `test_labels.pkl` → Test labels

---

## DAG 2: Data Preprocessing Pipeline

**Schedule**: `15 22 * * 0` (10:15 PM every Sunday)  
**Purpose**: Prepare data for sklearn and TensorFlow models

### Key Tasks

```python
def preprocess_data_task():
    # Set random seeds for reproducibility
    set_random_seeds(CONFIG)
    configure_gpu(CONFIG)
    
    # Prepare sklearn data (flattened)
    X_train_sk, y_train_sk = prepare_data_for_sklearn(train_images, train_labels)
    X_test_sk, y_test_sk = prepare_data_for_sklearn(test_images, test_labels)
    
    # Prepare tensorflow data (48x48x3)
    X_train_tf, y_train_tf = prepare_data_for_tensorflow(train_images, train_labels)
    X_test_tf, y_test_tf = prepare_data_for_tensorflow(test_images, test_labels)
    
    # Save preprocessed data
    save_preprocessed_data(...)
```

**Output:**
- `X_train_sk.pkl`, `y_train_sk.pkl` → Sklearn training data (flattened)
- `X_test_sk.pkl`, `y_test_sk.pkl` → Sklearn test data
- `X_train_tf.pkl`, `y_train_tf.pkl` → TensorFlow training data (48×48×3)
- `X_test_tf.pkl`, `y_test_tf.pkl` → TensorFlow test data

---

## DAG 3: Sklearn Models Training

**Schedule**: `45 22 * * 0` (10:45 PM every Sunday)  
**Purpose**: Train Logistic Regression and Feedforward Neural Network

### Training Process

```python
def train_sklearn_models_task():
    # Setup MLflow
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()
    
    # Train Logistic Regression
    if CONFIG.lr_enabled:
        with mlflow.start_run(run_name="logistic_regression"):
            lr_model = train_logistic_regression(X_train_sk, y_train_sk)
            
            # Evaluate model
            metrics = get_report(lr_model, X_train_sk, y_train_sk, 
                                X_test_sk, y_test_sk, model_type='sklearn')
            
            # Log to MLflow
            mlflow.log_params({...})
            mlflow.log_metrics({
                "train_accuracy": 89.57%,
                "test_accuracy": 87.52%,
                "test_f1": 89.15%,
                "test_roc_auc": 94.52%
            })
            mlflow.sklearn.log_model(lr_model, "LogisticRegression_EmotionDetection")
    
    # Train Feedforward NN
    if CONFIG.ffn_enabled:
        with mlflow.start_run(run_name="feedforward_neural_network"):
            ffn_model = train_ffn(X_train_sk, y_train_sk)
            # Similar logging process...
```

**Logged Metrics (Logistic Regression):**
- Train Accuracy: 73.35%
- Test Accuracy: 71.79%
- Test F1-Score: 77.18%
- Test Precision: 70.70%
- Test Recall: 84.98%
- Test ROC AUC: 79.31%

**Artifacts:**
- Confusion matrices
- ROC curves
- Metric comparison plots
- Trained models registered in MLflow

---

## DAG 4: CNN Model Training

**Schedule**: `45 22 * * 0` (10:45 PM every Sunday, parallel with DAG 3)  
**Purpose**: Train CNN model with optional K-Fold cross-validation

### Training Process

```python
def train_cnn_model_task():
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="cnn_model"):
        cnn_model = build_cnn()
        
        if CONFIG.cv_enabled:
            # K-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tf)):
                with mlflow.start_run(run_name=f"cnn_fold_{fold+1}", nested=True):
                    model_fold = build_cnn()
                    model_fold, history = train_cnn(
                        model=model_fold,
                        X_train=X_train_fold,
                        y_train=y_train_fold,
                        batch_size=32,
                        epochs=30,
                        augmentation_params=CONFIG.augmentation_params
                    )
                    # Log fold metrics...
            
            # Log average CV metrics
            mlflow.log_metrics({
                "avg_train_accuracy": ...,
                "avg_val_accuracy": ...
            })
        else:
            # Single training run
            cnn_model, history = train_cnn(...)
            metrics = get_report(cnn_model, X_train_tf, y_train_tf,
                                X_test_tf, y_test_tf, model_type="tensorflow")
            
            mlflow.log_metrics({
                "train_accuracy": 89.57%,
                "test_accuracy": 87.52%,
                "test_f1": 89.15%,
                "test_roc_auc": 94.52%
            })
            
            # Save and register model
            cnn_model.save(f"{CONFIG.models_dir}/CNN_EMOTION_DETECTION.keras")
            mlflow.tensorflow.log_model(cnn_model, "CNN_EmotionDetection")
```

**CNN Architecture:**
- Input: 48×48×3 (RGB images)
- Conv Layers: 4 layers (16→32→64→32 filters)
- Dense Layers: 64 units with dropout
- Output: Sigmoid activation (binary classification)
- Data Augmentation: Rotation, zoom, flips

**Logged Metrics (CNN - Best Model):**
- Train Accuracy: 89.57%
- Test Accuracy: 87.52%
- Test F1-Score: 89.15%
- Test Precision: 87.04%
- Test Recall: 91.37%
- Test ROC AUC: 94.52%

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│           EMOTION DETECTION PIPELINE FLOW                │
└─────────────────────────────────────────────────────────┘

[10:00 PM] DAG 1: DATA LOADING
├── Load train/test images from directories
├── Load labels from CSV files
├── Save: train_images.pkl, test_images.pkl, labels
└── Trigger → DAG 2

[10:15 PM] DAG 2: PREPROCESSING
├── Load raw images from DAG 1
├── Prepare sklearn data (flatten to 6912 features)
├── Prepare TensorFlow data (48×48×3 shape)
├── Normalize pixel values [0, 1]
├── Save: X_train_sk.pkl, X_train_tf.pkl, etc.
└── Trigger → DAG 3 & DAG 4 (parallel)

[10:45 PM] DAG 3: SKLEARN TRAINING (parallel)
├── Load sklearn data from DAG 2
├── Train Logistic Regression (71.79% acc)
│   ├── Evaluate on train & test
│   ├── Generate confusion matrix & ROC curve
│   ├── Log all metrics to MLflow
│   └── Register model in MLflow Registry
├── Train Feedforward NN (70.26% acc)
│   └── Similar logging process
└── End

[10:45 PM] DAG 4: CNN TRAINING (parallel)
├── Load TensorFlow data from DAG 2
├── Build CNN model
├── IF cross-validation enabled:
│   ├── Run 5-Fold CV
│   ├── Train model on each fold
│   ├── Log fold metrics
│   └── Log average metrics
├── ELSE:
│   ├── Train single CNN model (87.52% acc) ⭐
│   ├── Generate confusion matrix & ROC curve
│   ├── Log metrics to MLflow
│   ├── Save model locally (.keras format)
│   └── Register in MLflow Model Registry
└── End

[Result] Best model (CNN) ready for deployment