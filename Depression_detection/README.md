# Depression Detection

Our depression detection model leverages advanced natural language processing to analyze text data and identify patterns indicative of depression. This early detection system enables individuals to receive timely interventions and resources, helping them manage their condition and improve their quality of life.

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/7de96f15-a505-4841-878d-3214c45b5ed4

## Model Performance Comparison

We progressively improved our system through multiple generations:

### 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Multinomial Naive Bayes | 85.95% | 86.00% | 86.00% | 86.00% | 93.30% | ~19s |
| Stochastic Gradient Descent | 87.38% | 87.00% | 87.00% | 87.00% | 94.41% | ~33s |
| 🏆 RoBERTa + LoRA (Production) | **91.03%** | **91.49%** | **89.93%** | **90.70%** | **96.85%** | ~2 hours* |

*GPU-accelerated training on Tesla T4 with parameter-efficient fine-tuning

### Model Selection

We selected **RoBERTa + LoRA** as our production model due to:

✅ **Superior Performance**: 91.03% accuracy (+3.65% over SGD)  
✅ **Excellent AUC-ROC**: 96.85% for reliable discrimination  
✅ **Contextual Understanding**: Captures semantic meaning and context  
✅ **Transfer Learning**: Leverages pre-trained language knowledge  
✅ **Parameter Efficiency**: LoRA fine-tunes only 0.94% of parameters

### Key Improvements Over Traditional ML

**Accuracy**: +3.65% absolute improvement (91.03% vs 87.38%)  
**AUC-ROC**: +2.44% improvement (96.85% vs 94.41%)  
**False Positives**: 31.9% reduction (8.51% vs 12.50%)  
**False Negatives**: 22.5% fewer missed cases (10.07% vs 13.00%)

## Pipeline Overview

**Complete Pipeline:**
1. [Data loading](#dag-1-data-loading-pipeline)
2. [Text preprocessing](#dag-2-text-preprocessing-pipeline)
3. [Vectorization/Tokenization](#dag-3-vectorization-pipeline)
4. [Model training](#dag-4-model-training-pipeline)

## Dataset

**Total Samples**: 329,593 text entries

**Class Distribution:**
- Not Depressed: 169,174 (51.3%)
- Depressed: 160,419 (48.7%)

**Data Split:**
- Training: 70% (230,715)
- Validation: 15% (49,439)
- Test: 15% (49,439)

## Automated Training Pipeline (Apache Airflow)

Production-grade automated pipeline with 4 sequential DAGs running weekly on Sunday nights.

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────┐
│            WEEKLY TRAINING PIPELINE                   │
│             (Every Sunday Night)                      │
└──────────────────────────────────────────────────────┘

DAG 1: Data Loading         → 10:00 PM Sunday
  ↓ (Trigger)
DAG 2: Preprocessing        → 10:10 PM Sunday
  ↓ (Trigger)
DAG 3: Vectorization        → 10:40 PM Sunday
  ↓ (Trigger)
DAG 4: Model Training       → 11:00 PM Sunday
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
**Purpose**: Load and split depression detection dataset

### Key Tasks

```python
def load_data_task():
    # Load data
    train_data, test_data = load_data(
        raw_data_path=CONFIG.raw_data_path,
        target_column_names=CONFIG.target_columns,
        text_column_names=CONFIG.text_columns,
        test_size=0.3,
        random_state=42
    )
    
    # Save for next DAG
    save_to_pickle(train_data, test_data)
```

**Output:**
- `train_tweets.pkl` → Training data (70%)
- `test_tweets.pkl` → Test data (30%)

---

## DAG 2: Text Preprocessing Pipeline

**Schedule**: `10 22 * * 0` (10:10 PM every Sunday)  
**Purpose**: Clean and preprocess text data

### Key Tasks

```python
def preprocess_data_task():
    # Set random seeds
    set_random_seeds(CONFIG)
    
    # Load raw data
    train_tweets = load_pickle("train_tweets.pkl")
    test_tweets = load_pickle("test_tweets.pkl")
    
    # Clean tweets
    processed_train = []
    for tweet in train_tweets["filtered_tweet"]:
        processed = clean_tweets(tweet)
        processed_train.append(processed)
    
    # Save preprocessed data
    save_preprocessed(X_train, y_train, X_test, y_test)
```

**Preprocessing Steps:**
- Lowercase conversion
- URL/mention removal
- Special character handling
- Stopword removal
- Text normalization

**Output:**
- `X_train.pkl` → Clean texts (training)
- `y_train.pkl` → Labels (training)
- `X_test.pkl` → Clean texts (test)
- `y_test.pkl` → Labels (test)

---

## DAG 3: Vectorization Pipeline

**Schedule**: `40 22 * * 0` (10:40 PM every Sunday)  
**Purpose**: Transform text for Classical ML (TF-IDF) and BERT (tokenization)

### Parallel Processing

This DAG runs two tasks in parallel:

#### Task 3A: Classical ML Vectorization

```python
def vectorize_classical_ml_task():
    # Skip if classical ML disabled
    if not CONFIG.get_pipeline("classical_ml")["enabled"]:
        return
    
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="tfidf_vectorization"):
        # Build TF-IDF vectorizer
        vectorizer = build_tfidf_vectorizer()
        
        # Transform data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Log to MLflow
        mlflow.log_params({
            "tfidf_max_features": 30000,
            "vocabulary_size": len(vectorizer.vocabulary_)
        })
        
        # Save and register
        save_vectorizer(vectorizer)
        mlflow.sklearn.log_model(vectorizer, "TFIDF_Vectorizer_Depression")
```

**Output:**
- `X_train_vec.pkl` → TF-IDF matrix (training)
- `X_test_vec.pkl` → TF-IDF matrix (test)
- `tfidf_vectorizer.pkl` → Trained vectorizer
- Registered in MLflow Registry

#### Task 3B: BERT Tokenization

```python
def prepare_bert_datasets_task():
    # Skip if BERT disabled
    if not CONFIG.get_pipeline("bert")["enabled"]:
        return
    
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="bert_tokenization"):
        # Load tokenizer
        tokenizer = build_bert_tokenizer()
        
        # Split test into val/test
        X_val, X_test_final, y_val, y_test_final = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_bert_datasets(
            X_train, y_train,
            X_val, y_val,
            X_test_final, y_test_final,
            tokenizer
        )
        
        # Log to MLflow
        mlflow.log_params({
            "bert_model_name": "roberta-base",
            "bert_max_length": 128
        })
        
        # Save datasets and tokenizer
        save_bert_datasets(train_dataset, val_dataset, test_dataset)
        tokenizer.save_pretrained("bert_tokenizer")
```

**Output:**
- `bert_train_dataset.pkl` → Training dataset
- `bert_val_dataset.pkl` → Validation dataset
- `bert_test_dataset.pkl` → Test dataset
- `bert_tokenizer/` → Saved tokenizer

---

## DAG 4: Model Training Pipeline

**Schedule**: `0 23 * * 0` (11:00 PM every Sunday)  
**Purpose**: Train Classical ML models

### Training Process

```python
def train_data_task():
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    # Load vectorized data
    X_train = load_pickle("X_train_vec.pkl")
    X_test = load_pickle("X_test_vec.pkl")
    
    # Train Multinomial Naive Bayes
    if CONFIG.mnb_enabled:
        with mlflow.start_run(run_name="multinomial_nb"):
            mnb_model = train_multinomial_nb(X_train, y_train)
            
            # Cross-validation
            if CONFIG.cv_enabled:
                cv_scores = run_cv_sklearn(mnb_model, X_train, y_train)
                mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
            
            # Evaluate
            results = get_report(mnb_model, X_train, y_train, X_test, y_test)
            
            mlflow.log_metrics({
                "train_accuracy": results["train"]["accuracy"],
                "test_accuracy": results["test"]["accuracy"],
                "test_f1": results["test"]["f1"],
                "test_auc": results["test"]["roc_auc"]
            })
            
            # Register model
            mlflow.sklearn.log_model(mnb_model, "Depression_Detection_MNB")
    
    # Train SGD Classifier
    if CONFIG.sgd_enabled:
        with mlflow.start_run(run_name="sgd_classifier"):
            sgd_model = train_sgd_classifier(X_train, y_train)
            # Similar logging process...
```

**Logged Metrics (Classical ML):**

| Model | Train Acc | Test Acc | F1 | Precision | Recall | AUC-ROC |
|-------|-----------|----------|-----|-----------|--------|---------|
| **MNB** | 85.95% | 85.95% | 86.00% | 86.00% | 86.00% | 93.30% |
| **SGD** | 87.38% | 87.38% | 87.00% | 87.00% | 87.00% | 94.41% |

**Artifacts:**
- Confusion matrices
- ROC curves
- Metric plots
- Trained models registered in MLflow

---

## RoBERTa + LoRA Model (Production)

### Model Architecture

```
Input Text (max 128 tokens)
         ↓
    Tokenization
         ↓
┌─────────────────────────┐
│   RoBERTa-base Encoder  │
│   (12 Transformer Layers)│
│   + LoRA Adapters       │
│     Rank: 16, Alpha: 32 │
└─────────────────────────┘
         ↓
   Classification Head
         ↓
Output: [P(Not Depressed), P(Depressed)]
```

**Specifications:**
- Total Parameters: 125,829,124
- Trainable (LoRA): 1,181,954 (0.94%)
- Hidden Size: 768
- Attention Heads: 12
- Max Length: 128 tokens

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=128,
    learning_rate=auto,
    fp16=True,
    eval_strategy='steps',
    load_best_model_at_end=True
)

LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1
)
```

**Results:**
- Training Time: ~2 hours (Tesla T4)
- Best Epoch: 5 (early stopping)
- Validation Loss: 0.2194
- No overfitting observed

---

## Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────┐
│        DEPRESSION DETECTION PIPELINE FLOW             │
└──────────────────────────────────────────────────────┘

[10:00 PM] DAG 1: DATA LOADING
├── Load depression dataset (329K texts)
├── Split 70/30 (train/test)
├── Save: train_tweets.pkl, test_tweets.pkl
└── Trigger → DAG 2

[10:10 PM] DAG 2: PREPROCESSING
├── Load raw data from DAG 1
├── Clean tweets (lowercase, remove URLs, etc.)
├── Save: X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl
└── Trigger → DAG 3

[10:40 PM] DAG 3: VECTORIZATION (parallel)
├── Task 3A: Classical ML
│   ├── Build TF-IDF vectorizer (30K features)
│   ├── Transform train & test
│   ├── Save: X_train_vec.pkl, X_test_vec.pkl
│   └── Register vectorizer in MLflow
│
├── Task 3B: BERT Tokenization
│   ├── Load RoBERTa tokenizer
│   ├── Split test → val/test
│   ├── Create tokenized datasets
│   └── Save: bert_train/val/test_dataset.pkl
│
└── Trigger → DAG 4

[11:00 PM] DAG 4: MODEL TRAINING
├── Train Classical ML models:
│   ├── Multinomial NB (85.95% acc)
│   └── SGD Classifier (87.38% acc)
├── For each model:
│   ├── Cross-validation (optional)
│   ├── Evaluate on train & test
│   ├── Generate confusion matrix & ROC curve
│   ├── Log metrics to MLflow
│   └── Register in MLflow Registry
└── End

[Separate] RoBERTa Training (Jupyter/Colab)
├── Load BERT datasets
├── Fine-tune RoBERTa + LoRA
├── Achieve 91.03% accuracy ⭐
└── Deploy to production
```

---

## Usage

### RoBERTa Model Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = 'ziadashraf98765/roberta-depression-detection-lora-merged'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict
def predict_depression(text):
    encoding = tokenizer(text, max_length=128, truncation=True, 
                        padding='max_length', return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return {
        'prediction': 'Depressed' if pred == 1 else 'Not Depressed',
        'confidence': probs[0][pred].item()
    }

# Example
result = predict_depression("I feel hopeless and empty inside")
print(f"{result['prediction']} (Confidence: {result['confidence']:.2%})")
```

---

## MLflow Integration

Full experiment tracking for all pipeline stages:

**Tracked Information:**
- ✅ Parameters: Hyperparameters, vectorizer settings
- ✅ Metrics: Accuracy, precision, recall, F1, AUC-ROC
- ✅ Artifacts: Models, vectorizers, tokenizers, plots
- ✅ System Metrics: CPU, memory, GPU usage
- ✅ Model Registry: Versioned models with metadata

---

## Source Code Structure

```
Depression_detection/
├── src/
│   ├── config/
│   │   ├── train_config_loader.py
│   │   └── train_config_random_seed.py
│   ├── logger/
│   │   └── train_logger.py
│   ├── text/
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── models/
│   │   ├── model_vectorizer.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   └── utils/
├── notebooks/
│   ├── Models_Dev.ipynb
│   └── Depression_Detection_With_Bert.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── plots/
├── logs/
└── train_config.yaml
```

---

## Clinical Applications

### Use Cases
- Early screening for at-risk individuals
- Social media mental health monitoring
- Crisis intervention flagging
- Treatment progress tracking
- Research & epidemiology

### Clinical Metrics (RoBERTa)
- **Sensitivity**: 89.93% - Identifies 9/10 depression cases
- **Specificity**: 92.07% - Correctly identifies 92/100 non-depression cases
- **PPV (Precision)**: 91.49% - 91% of positive predictions are correct
- **NPV**: 90.51% - 91% of negative predictions are correct

### ⚠️ Important Note

**This is a screening tool, NOT a diagnostic instrument:**
- ✅ Use as first-line screening aid
- ✅ Flag high-risk individuals for evaluation
- ✅ Support clinical decision-making
- ❌ Do NOT use as sole diagnosis basis
- ❌ Always involve qualified professionals

---

**Model Version**: 2.0 (RoBERTa + LoRA)  
**Previous Version**: 1.0 (Ensemble: SGD + Naive Bayes)  
**Last Updated**: 2025