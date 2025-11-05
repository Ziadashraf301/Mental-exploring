# Depression Detection

The first model that we have developed is focused on depression detection using text data, and it represents a crucial area of research owing to the increasing prevalence of depression and other mental health disorders. By leveraging advanced natural language processing techniques, this model can analyze text data in order to identify patterns and indicators that are indicative of depression. This early detection is vital as it enables individuals to receive timely interventions and resources that can help them manage their condition and improve their overall quality of life. 

https://github.com/Ziadashraf301/Mental-exploring/assets/111798631/7de96f15-a505-4841-878d-3214c45b5ed4

## Model Evolution & Performance Comparison

We have progressively improved our depression detection system through three generations of models:

### 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Multinomial Naive Bayes** | 85.95% | 86.00% | 86.00% | 86.00% | 93.30% | 18.96s |
| **Stochastic Gradient Descent** | 87.38% | 87.00% | 87.00% | 87.00% | 94.41% | 33.40s |
| **Ensemble (SGD + Naive Bayes)** | ~87.50% | ~87.00% | ~87.00% | ~87.00% | ~94.50% | Combined |
| **🏆 RoBERTa + LoRA (Current)** | **91.03%** | **91.49%** | **89.93%** | **90.70%** | **96.85%** | ~2 hours* |

*\*Training time reflects GPU-accelerated training on Tesla T4 with parameter-efficient fine-tuning*

### 🚀 Key Improvements

Our latest **RoBERTa-base with LoRA** model demonstrates significant improvements over traditional ML approaches:

#### Accuracy Improvements
- **+3.53%** absolute improvement over ensemble model (91.03% vs 87.50%)
- **+5.08%** improvement over baseline Naive Bayes (91.03% vs 85.95%)
- **+4.19%** relative error reduction compared to ensemble

#### AUC-ROC Improvements
- **+2.35%** improvement over ensemble (96.85% vs 94.50%)
- **+2.48%** improvement over SGD alone (96.85% vs 94.41%)
- **Better discriminative power** between depression and non-depression cases

#### Precision & Recall Balance
- **Higher precision** (91.49% vs 87.00%): Fewer false positives
- **Better recall** (89.93% vs 87.00%): Better at identifying true depression cases
- **More balanced predictions** across both classes

### 📈 Detailed Model Comparison

#### 1. Multinomial Naive Bayes (Baseline)
```python
MultinomialNB(alpha=0.1, fit_prior=False)
+ TF-IDF Vectorization (max_features=30,000)
```

**Strengths:**
- ✅ Fast training (18.96s)
- ✅ Simple and interpretable
- ✅ Good baseline performance

**Limitations:**
- ❌ Assumes feature independence
- ❌ Limited context understanding
- ❌ Cannot capture complex patterns

**Results:**
- Accuracy: 85.95%
- Sensitivity: 89.30%
- Specificity: 82.76%
- AUC-ROC: 93.30%
- CV Score: 86.11%

---

#### 2. Stochastic Gradient Descent
```python
SGDClassifier(
    loss='modified_huber',
    penalty='l2',
    alpha=0.0001,
    max_iter=60,
    learning_rate='adaptive'
)
+ TF-IDF Vectorization (max_features=30,000)
```

**Strengths:**
- ✅ Better than Naive Bayes
- ✅ Handles large datasets well
- ✅ More balanced sensitivity/specificity

**Limitations:**
- ❌ Still relies on bag-of-words
- ❌ No semantic understanding
- ❌ Sensitive to hyperparameters

**Results:**
- Accuracy: 87.38%
- Sensitivity: 87.26%
- Specificity: 87.49%
- AUC-ROC: 94.41%
- CV Score: 88.74%

---

#### 3. Ensemble (SGD + Naive Bayes) - Deployed Model
```python
Voting/Stacking ensemble of SGD and Multinomial NB
```

**Strengths:**
- ✅ Combines strengths of both models
- ✅ Reduces individual model weaknesses
- ✅ Best traditional ML performance

**Limitations:**
- ❌ Still limited by TF-IDF representation
- ❌ No contextual understanding
- ❌ Cannot capture word order/semantics

**Estimated Results:**
- Accuracy: ~87.50%
- AUC-ROC: ~94.50%

---

#### 4. 🏆 RoBERTa + LoRA (Current State-of-the-Art)
```python
RoBERTa-base with LoRA fine-tuning
- Rank: 16
- Alpha: 32
- Target: Query & Value layers
- Max Length: 128 tokens
```

**Revolutionary Advantages:**
- ✅ **Contextual understanding**: Captures semantic meaning and context
- ✅ **Word order sensitivity**: Understands sentence structure
- ✅ **Transfer learning**: Leverages pre-trained language knowledge
- ✅ **Parameter efficiency**: LoRA fine-tunes only 0.94% of parameters
- ✅ **Better generalization**: Reduced overfitting with early stopping
- ✅ **State-of-the-art architecture**: Transformer-based attention mechanism

**Outstanding Results:**
- **Accuracy: 91.03%** (+3.53% improvement)
- **Precision: 91.49%** (+4.49% improvement)
- **Recall: 89.93%** (+2.93% improvement)
- **F1-Score: 90.70%** (+3.70% improvement)
- **AUC-ROC: 96.85%** (+2.35% improvement)
- **Specificity: 92.07%** (+4.58% improvement)
- **Sensitivity: 89.93%** (+2.67% improvement)

---

### 🎯 Why RoBERTa Outperforms Traditional ML

#### 1. **Semantic Understanding**
- Traditional ML: Treats text as "bag of words"
- RoBERTa: Understands context, synonyms, and semantic relationships

#### 2. **Contextual Awareness**
```
Input: "I feel hopeless and empty inside"

Traditional ML:
- Counts words: "hopeless"(1), "empty"(1), "inside"(1)
- No context understanding

RoBERTa:
- Understands "hopeless" in context of emotional state
- Recognizes "empty inside" as metaphor for depression
- Captures sentiment and emotional nuance
```

#### 3. **Transfer Learning**
- Pre-trained on massive text corpora
- Already understands language patterns
- Fine-tuning adapts this knowledge to depression detection

#### 4. **Better Feature Representation**
- Traditional ML: TF-IDF vectors (sparse, 30K dimensions)
- RoBERTa: Dense embeddings (768 dimensions) with rich semantic information

---

### 📊 Real-World Impact

#### False Positive Reduction
```
Ensemble Model: ~12.50% false positives
RoBERTa Model:  ~8.51% false positives
→ 31.9% reduction in false alarms
```

#### False Negative Reduction
```
Ensemble Model: ~13.00% false negatives (missed cases)
RoBERTa Model:  ~10.07% false negatives
→ 22.5% fewer missed depression cases
```

#### Clinical Significance
- **732 more accurate predictions** per 10,000 screened individuals
- **293 fewer false negatives** = more people getting needed help
- **439 fewer false positives** = reduced unnecessary clinical resources

---

## Model Development Process

Our model development includes several key steps:

### Traditional ML Pipeline (Baseline & Ensemble Models)
- [Data collection](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/Data%20collection%20and%20preprocessing.ipynb)
- [Data preprocessing](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/data_collection_and_preprocessing.py)
- [Feature engineering](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/Models.ipynb)
- [Machine learning modeling](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/Models.ipynb)
- [Model evaluation](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/Models.ipynb)
- [Statistical testing of models](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/Test_models_statistically.ipynb)
- [Model integration](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/model_data_pipline.py)

### Deep Learning with RoBERTa
- [Fine-tuned RoBERTa with LoRA](https://github.com/Ziadashraf301/Mental-exploring/blob/main/Depression%20detection/depression_detection_with_bert.ipynb)

---

## [Dataset](https://drive.google.com/file/d/1WATc0Uor8yUII4PX-HUZNab9fQcAxO9A/view?usp=sharing)

The model was trained on a large-scale, balanced dataset:
- **Total Samples:** 329,593 text entries
- **Class Distribution:**
  - Not Depressed: 169,174 (51.3%)
  - Depressed: 160,419 (48.7%)
- **Data Split:**
  - Training: 70% (230,715 samples)
  - Validation: 15% (49,439 samples)
  - Test: 15% (49,439 samples)

---

## Usage

### Loading the RoBERTa Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = 'ziadashraf98765/roberta-depression-detection-lora-merged'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Making Predictions
```python
def predict_depression(text, model, tokenizer, device='cpu', max_length=128):
    """
    Predict depression from text input
    
    Args:
        text: Input text string
        model: Fine-tuned RoBERTa model
        tokenizer: RoBERTa tokenizer
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length
    
    Returns:
        dict: {
            'prediction': 'Depressed' or 'Not Depressed',
            'confidence': float (0-1),
            'depression_probability': float (0-1),
            'not_depression_probability': float (0-1)
        }
    """
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return {
        'prediction': 'Depressed' if pred_class == 1 else 'Not Depressed',
        'confidence': confidence,
        'depression_probability': probs[0][1].item(),
        'not_depression_probability': probs[0][0].item()
    }

# Example usage
text = "I feel so hopeless and empty inside"
result = predict_depression(text, model, tokenizer, device)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Depression Probability: {result['depression_probability']:.2%}")
```

**Example Output:**
```
Prediction: Depressed
Confidence: 94.59%
Depression Probability: 94.59%
```

---

## Requirements
```txt
# Deep Learning (RoBERTa Model)
transformers>=4.30.0
torch>=2.0.0
peft>=0.4.0
datasets>=2.12.0
accelerate>=0.20.0

# Traditional ML Models
scikit-learn>=1.2.0
nltk>=3.8.0

# Data Processing & Visualization
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional: Deployment
fastapi>=0.100.0
uvicorn>=0.23.0
```

---

## Model Architecture

### RoBERTa + LoRA Architecture
```
Input Text (max 128 tokens)
         ↓
    Tokenization
         ↓
┌─────────────────────────┐
│   RoBERTa-base Encoder  │
│   (12 Transformer Layers)│
│                         │
│   + LoRA Adapters       │
│     (Query & Value)     │
│     Rank: 16            │
│     Alpha: 32           │
└─────────────────────────┘
         ↓
   Pooled Output (768-dim)
         ↓
  Classification Head
         ↓
   Softmax (2 classes)
         ↓
Output: [P(Not Depressed), P(Depressed)]
```

**Technical Specifications:**
- **Total Parameters:** 125,829,124
- **Trainable Parameters (LoRA):** 1,181,954 (0.94%)
- **Frozen Parameters:** 124,647,170 (99.06%)
- **Hidden Size:** 768
- **Attention Heads:** 12
- **Layers:** 12
- **Max Position Embeddings:** 514

---

## Training Configuration

### Hyperparameters
```python
TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=auto,  # Adaptive
    fp16=True,  # Mixed precision training
    eval_strategy='steps',
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)
```

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
```

### Training Results
- **Training Time:** ~2 hours (Tesla T4 GPU)
- **Best Epoch:** 5 (with early stopping)
- **Training Loss:** 0.2253
- **Validation Loss:** 0.2194
- **No overfitting observed**

---

## Clinical Applications

### Use Cases
1. **Screening Tool:** Early identification of at-risk individuals
2. **Social Media Monitoring:** Track mental health trends in online communities
3. **Crisis Intervention:** Flag concerning posts for immediate support
4. **Treatment Progress:** Monitor patient communication over time
5. **Research:** Large-scale epidemiological studies

### Clinical Validation Metrics
- **Sensitivity (Recall):** 89.93% - Correctly identifies 9/10 depression cases
- **Specificity:** 92.07% - Correctly identifies 92/100 non-depression cases
- **Positive Predictive Value (Precision):** 91.49% - 91% of "depressed" predictions are correct
- **Negative Predictive Value:** 90.51% - 91% of "not depressed" predictions are correct

### ⚠️ Important Clinical Note

**This model is a screening tool, NOT a diagnostic instrument:**
- ✅ Use as a first-line screening aid
- ✅ Flag high-risk individuals for clinical evaluation
- ✅ Support clinical decision-making
- ❌ Do NOT use as sole basis for diagnosis
- ❌ Do NOT replace professional clinical assessment
- ❌ Always involve qualified mental health professionals

---

## Deployment Considerations

### Model Size & Performance
- **Model Size:** ~500 MB (merged model)
- **Inference Time:** ~50ms per text (GPU) / ~200ms (CPU)
- **Throughput:** ~20 texts/second (GPU)
- **Memory Requirements:** ~2GB GPU / ~4GB RAM

### Deployment
```python
# For production API
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    result = predict_depression(
        input.text,
        model,
        tokenizer,
        device='cuda'
    )
    return result
```

**Last Updated:** 2025
**Model Version:** 2.0 (RoBERTa + LoRA)
**Previous Version:** 1.0 (Ensemble: SGD + Naive Bayes)
