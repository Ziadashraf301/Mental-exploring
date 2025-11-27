from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Depression_detection.src.logger.train_logger import get_logger
from Depression_detection.src.config.train_config_loader import get_train_config

LOGGER = get_logger()

# CLASSICAL ML: TF-IDF VECTORIZATION
def build_tfidf_vectorizer():
    """
    Build and return a TF-IDF vectorizer pipeline based on configuration settings.
    """
    config = get_train_config()
    tfidf_cfg = config.pipelines["classical_ml"]["tf_idf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        stop_words=tfidf_cfg["stop_words"]
    )

    LOGGER.info(f"TF-IDF Vectorizer created with params: max_features={tfidf_cfg['max_features']}, ngram_range={tfidf_cfg['ngram_range']}, stop_words={tfidf_cfg['stop_words']}")
    return vectorizer

def transform_data(vectorizer, X_train, X_test):
    """
    Fit the TF-IDF vectorizer on training data and transform both training and test data.
    """
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    LOGGER.info("Data transformed using TF-IDF vectorizer")
    return X_train_tfidf, X_test_tfidf


# BERT: TOKENIZATION & DATASET
class BERTDepressionDataset(Dataset):
    """
    PyTorch Dataset for BERT-based depression detection.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_bert_tokenizer():
    """
    Build and return BERT tokenizer based on configuration settings.
    """
    config = get_train_config()
    bert_cfg = config.pipelines["bert"]["tokenization"]

    model_name = bert_cfg.get("model_name", "roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    LOGGER.info(f"BERT Tokenizer loaded: {model_name}")
    return tokenizer


def create_bert_datasets(X_train, y_train, X_val, y_val, X_test, y_test, tokenizer):
    """
    Create PyTorch datasets for BERT training, validation, and testing.
    """
    config = get_train_config()
    bert_cfg = config.pipelines["bert"]["tokenization"]
    max_length = bert_cfg.get("max_length", 128)
    
    train_dataset = BERTDepressionDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = BERTDepressionDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = BERTDepressionDataset(X_test, y_test, tokenizer, max_length)
    
    LOGGER.info(f"BERT datasets created with max_length={max_length}")
    LOGGER.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset