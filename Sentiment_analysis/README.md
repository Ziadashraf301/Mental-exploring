# Sentiment Analysis

Our sentiment analysis model is designed to understand people's emotions and attitudes toward specific topics or products. Using natural language processing and machine learning techniques, it analyzes text data such as customer reviews or social media posts to determine the overall sentiment expressed.

In fields like marketing and customer service, sentiment analysis plays a crucial role in improving customer experiences and informing business decisions. By analyzing customer feedback and identifying positive or negative sentiment, businesses can gain valuable insights into customer preferences and use this information to improve their products and services. Additionally, sentiment analysis helps businesses identify and address customer complaints in a timely manner, improving customer satisfaction and retention.

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/95c7c048-5ea8-46b9-919b-2d4a3ad85037

## Model Performance Comparison

We developed and evaluated three machine learning models for sentiment analysis:

### 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Multinomial Naive Bayes | 77.01% | 76.93% | 77.16% | 77.04% | 85.18% | ~30s |
| 🏆 Logistic Regression (Production)| 78.33% | 77.94% | 79.03% | 78.48% | 86.44% | ~2 min |
| Linear SVC | 78.12% | 77.52% | 79.22% | 78.36% | 86.30% | ~2 min |

### Model Selection

We selected **Logistic Regression** as our production model due to:

✅ **Strong Performance**: 78.33% accuracy with excellent balance  
✅ **Best ROC AUC**: 86.44% for reliable confidence scoring  
✅ **Interpretability**: Easy to understand feature importance  
✅ **Balanced Metrics**: Consistent precision (77.94%) and recall (79.03%)  
✅ **Production Ready**: Reliable and maintainable in deployment

### Confusion Matrix Analysis

Our Logistic Regression model shows balanced performance:

```
                 Predicted
                Neg     Pos
Actual  Neg    38.81%  11.19%
        Pos    10.48%  39.52%
```

**Interpretation:**
- True Negatives: 38.81% (correctly identified negative tweets)
- False Positives: 11.19% (negative tweets misclassified as positive)
- False Negatives: 10.48% (positive tweets misclassified as negative)
- True Positives: 39.52% (correctly identified positive tweets)

## Pipeline Overview

**Complete Pipeline:**
1. [Data loading](#dag-1-data-loading-pipeline)
2. [Text preprocessing](#dag-2-text-preprocessing-pipeline)
3. [TF-IDF vectorization](#dag-3-tf-idf-vectorization-pipeline)
4. [Model training](#dag-4-model-training-pipeline)

## Dataset

**Dataset**: Sentiment140 containing 1,600,000 tweets

**Class Distribution:**
- Negative (0): 800,000 tweets (50%)
- Positive (1): 800,000 tweets (50%)

**Data Split:**
- Training: 90% (1,440,000 tweets)
- Testing: 10% (160,000 tweets)

## Text Preprocessing Pipeline

Our 9-step preprocessing pipeline:

1. **Lower Casing** → "GREAT Product!" → "great product!"
2. **URL Replacement** → Remove http/https/www links
3. **Emoji Replacement** → ":)" → "smile"
4. **Username Removal** → "@user thanks" → "thanks"
5. **Non-Alphabet Removal** → "test#123" → "test 123"
6. **Consecutive Letter Reduction** → "Heyyyy" → "Heyy"
7. **Short Word Removal** → Remove words < 2 chars
8. **Stopword Removal** → "the cat is here" → "cat here"
9. **Lemmatization** → "running" → "run"

## Feature Engineering - TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text to numerical features:
- **TF**: Word frequency in document
- **IDF**: Word importance across all documents
- **Configuration**: Max 50,000 features, sparse matrix

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
DAG 3: TF-IDF Vectorization → 10:40 PM Sunday
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
**Purpose**: Load and split raw tweets dataset

### Key Tasks

```python
def load_data_task():
    # Load 1.6M tweets
    train_tweets, test_tweets = load_data(
        data_path=CONFIG.raw_data_path,
        test_size=0.1,  # 90% train, 10% test
        random_state=42
    )
    
    # Save for next DAG
    save_to_pickle(train_tweets, test_tweets)
```

**Output:**
- `train_tweets.pkl` → 1,440,000 tweets
- `test_tweets.pkl` → 160,000 tweets

---

## DAG 2: Text Preprocessing Pipeline

**Schedule**: `10 22 * * 0` (10:10 PM every Sunday)  
**Purpose**: Clean and preprocess tweet text

### Key Tasks

```python
def preprocess_data_task():
    # Load raw data from DAG 1
    train_tweets = load_pickle("train_tweets.pkl")
    test_tweets = load_pickle("test_tweets.pkl")
    
    # Apply 9-step preprocessing
    processed_train_text = []
    for tweet in train_tweets["text"]:
        processed = preprocess_text(tweet)
        processed_train_text.append(processed)
    
    # Save preprocessed data
    save_preprocessed(X_train, y_train, X_test, y_test)
```

**Output:**
- `X_train.pkl` → Clean texts (1.44M)
- `y_train.pkl` → Labels
- `X_test.pkl` → Clean texts (160K)
- `y_test.pkl` → Labels

---

## DAG 3: TF-IDF Vectorization Pipeline

**Schedule**: `40 22 * * 0` (10:40 PM every Sunday)  
**Purpose**: Transform text into TF-IDF feature vectors

### Key Tasks

```python
def vectorize_data_task():
    # MLflow setup
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="tfidf_vectorization"):
        # Build TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=50000)
        
        # Fit on training data only
        vectorizer.fit(X_train)
        
        # Transform both datasets
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Log to MLflow
        mlflow.log_params({
            "tfidf_max_features": 50000,
            "vocabulary_size": len(vectorizer.vocabulary_)
        })
        
        # Save and register vectorizer
        mlflow.sklearn.log_model(
            vectorizer,
            "TFIDF_Vectorizer",
            registered_model_name="TFIDF_Vectorizer_Sentiment"
        )
```

**Output:**
- `X_train_vec.pkl` → TF-IDF matrix (1.44M × 50K)
- `X_test_vec.pkl` → TF-IDF matrix (160K × 50K)
- `tfidf_vectorizer.pkl` → Trained vectorizer
- Registered in MLflow Model Registry

---

## DAG 4: Model Training Pipeline

**Schedule**: `0 23 * * 0` (11:00 PM every Sunday)  
**Purpose**: Train and evaluate all models, register in MLflow

### Training Process

```python
def train_data_task():
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    # Train Multinomial Naive Bayes
    if CONFIG.mnb_enabled:
        with mlflow.start_run(run_name="MultinomialNB"):
            mnb_model = train_multinomial_nb(X_train, y_train)
            
            metrics = get_report(mnb_model, X_train, y_train, 
                                X_test, y_test)
            
            mlflow.log_metrics({
                "train_accuracy": 77.99%,
                "test_accuracy": 77.01%,
                "test_f1": 77.04%,
                "test_roc_auc": 85.18%
            })
            mlflow.sklearn.log_model(mnb_model, "MultinomialNB_Sentiment")
    
    # Train Linear SVC
    if CONFIG.svc_enabled:
        with mlflow.start_run(run_name="LinearSVC"):
            svc_model = train_linear_svc(X_train, y_train)
            
            mlflow.log_metrics({
                "train_accuracy": 79.28%,
                "test_accuracy": 78.33%,
                "test_f1": 78.48%,
                "test_roc_auc": 86.44%
            })
            mlflow.sklearn.log_model(svc_model, "LinearSVC_Sentiment")
    
    # Train Logistic Regression (Best Model)
    if CONFIG.lr_enabled:
        with mlflow.start_run(run_name="LogisticRegression"):
            lr_model = train_logistic_regression(X_train, y_train)
            
            mlflow.log_metrics({
                "train_accuracy": 80.11%,
                "test_accuracy": 78.12%,
                "test_f1": 78.36%,
                "test_roc_auc": 86.30%
            })
            mlflow.sklearn.log_model(lr_model, "LogisticRegression_Sentiment")
```

**Logged Metrics:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Multinomial Naive Bayes | 77.01% | 76.93% | 77.16% | 77.04% | 85.18% | ~30s |
| 🏆 Logistic Regression (Production)  | 78.33% | 77.94% | 79.03% | 78.48% | 86.44% | ~2 min |
| Linear SVC| 78.12% | 77.52% | 79.22% | 78.36% | 86.30% | ~2 min |


**Artifacts:**
- Confusion matrices
- ROC curves
- Metric comparison plots
- Trained models registered in MLflow

---

## Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────┐
│          SENTIMENT ANALYSIS PIPELINE FLOW             │
└──────────────────────────────────────────────────────┘

[10:00 PM] DAG 1: DATA LOADING
├── Load sentiment140.csv (1.6M tweets)
├── Split 90/10 (train/test)
├── Save: train_tweets.pkl, test_tweets.pkl
└── Trigger → DAG 2

[10:10 PM] DAG 2: PREPROCESSING
├── Load raw tweets from DAG 1
├── Apply 9-step preprocessing:
│   ├── Lowercase, remove URLs/mentions
│   ├── Replace emojis
│   ├── Remove stopwords & lemmatize
├── Save: X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl
└── Trigger → DAG 3

[10:40 PM] DAG 3: TF-IDF VECTORIZATION
├── Load preprocessed text from DAG 2
├── Build TF-IDF vectorizer (max 50K features)
├── Fit on training, transform train & test
├── Log parameters to MLflow
├── Save: X_train_vec.pkl, X_test_vec.pkl
├── Register vectorizer in MLflow Registry
└── Trigger → DAG 4

[11:00 PM] DAG 4: MODEL TRAINING
├── Load vectorized data from DAG 3
├── Train models (parallel MLflow runs):
│   ├── Multinomial NB (77.01% acc)
│   ├── Linear SVC (78.33% acc)
│   └── Logistic Regression (78.12% acc) ⭐
├── For each model:
│   ├── Evaluate on train & test
│   ├── Generate confusion matrix & ROC curve
│   ├── Log all metrics to MLflow
│   └── Register in MLflow Model Registry
└── End

[Result] Best model (Logistic Regression) ready for deployment
```

---

## MLflow Integration

Full experiment tracking for every pipeline stage:

**Tracked Information:**
- ✅ **Parameters**: Hyperparameters, vectorizer settings
- ✅ **Metrics**: Accuracy, precision, recall, F1, ROC AUC
- ✅ **Artifacts**: Models, vectorizers, plots
- ✅ **System Metrics**: CPU, memory usage
- ✅ **Model Registry**: Versioned models with metadata

---

## Configuration Management

Centralized settings in `train_config.yaml`:

```yaml
# DATA PATHS
data:
  data_path: "Sentiment_analysis/data/raw/training.1600000.processed.noemoticon.csv"
  processed_data_path: "Sentiment_analysis/data/processed"

# MLFLOW
mlflow:
  experiment_name: "SentimentAnalysis_v2"
  tracking_uri: ""

# PREPROCESSING
preprocessing:
  tf_idf:
    max_features: 50000
    ngram_range: [1, 2]

# MODELS
logistic_regression:
  enabled: true
  max_iter: 300
  C: 1.0
  n_jobs: -1

# CROSS-VALIDATION
cross_validation:
  enabled: false
  k_folds: 5
```

---

## Source Code Structure

```
Sentiment_analysis/
├── src/
│   ├── config/
│   │   ├── train_config_loader.py
│   │   └── train_config_random_seed.py
│   ├── logger/
│   │   └── train_logger.py
│   ├── tweets/
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── models/
│   │   ├── model_vectorizer.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   └── utils/
│       └── helpers.py
├── notebooks/
│   ├── Models_Dev.ipynb
│   └── test_models_statistically.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── plots/
├── logs/
└── train_config.yaml
```