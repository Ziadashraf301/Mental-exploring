from sklearn.feature_extraction.text import TfidfVectorizer
from Sentiment_analysis.src.logger.train_logger import get_logger
from Sentiment_analysis.src.config.train_config_loader import get_train_config

LOGGER = get_logger()

def build_tfidf_vectorizer():
    """
    Build and return a TF-IDF vectorizer pipeline based on configuration settings.
    """
    config = get_train_config()
    tfidf_cfg = config.tfidf_params
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        stop_words=tfidf_cfg["stop_words"],
        min_df=tfidf_cfg["min_df"]
    )

    LOGGER.info(f"TF-IDF Vectorizer created with params: max_features={tfidf_cfg['max_features']}, ngram_range={tfidf_cfg['ngram_range']}, stop_words={tfidf_cfg['stop_words']}, min_df={tfidf_cfg['min_df']}")
    return vectorizer

def transform_data(vectorizer, X_train, X_test):
    """
    Fit the TF-IDF vectorizer on training data and transform both training and test data.
    """
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    LOGGER.info("Data transformed using TF-IDF vectorizer")
    return X_train_tfidf, X_test_tfidf