from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from Depression_detection.src.logger.train_logger import get_logger
from Depression_detection.src.config.train_config_loader import get_train_config

LOGGER = get_logger()

# MULTINOMIAL NAIVE BAYES
def train_multinomial_nb(X, y):
    """
    Train Multinomial Naive Bayes using config parameters.
    """
    config = get_train_config()
    params = config.pipelines["classical_ml"]["models"]["multinomial_nb"]

    model = MultinomialNB(
        alpha=params.get("alpha", 1.0),
        fit_prior=params.get("fit_prior", False),
        force_alpha=params.get("force_alpha", True)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained MultinomialNB with params: {params}")

    return model


# SGD Classifier
def train_sgd_classifier(X, y):
    """
    Train Linear SVC using config parameters.
    """
    config = get_train_config()
    params = config.pipelines["classical_ml"]["models"]["sgd_classifier"]

    model = SGDClassifier(
        loss=params.get("loss", "modified_huber"),
        penalty=params.get("penalty", "l2"),
        alpha=params.get("alpha", 0.0001),
        max_iter=params.get("max_iter", 60),
        tol=params.get("tol", None),
        learning_rate=params.get("learning_rate", "adaptive"),
        eta0=params.get("eta0", 0.01),
        fit_intercept=params.get("fit_intercept", False),
        random_state=config.random_seeds.get("numpy_seed", 42)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained SGD Classifier with params: {params}")

    return model


def run_cv_sklearn(model, X, y, scoring: str):
    """Run K-Fold CV only if enabled in config."""
    
    config = get_train_config()
    
    LOGGER.info(
        f"Running {config.cv_params.get('k_folds', 5)}-fold cross-validation "
        f"(shuffle={config.cv_params.get('shuffle', True)}, random_state={config.cv_params.get('random_state', 42)})"
        f" with scoring='{scoring}'"
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        shuffle=config.cv_params.get("shuffle", True),
        random_state=config.cv_params.get("random_state", 42),
        cv=config.cv_params.get("k_folds", 5),
        scoring=scoring,
        n_jobs=config.cv_params.get("n_jobs", -1)
    )

    LOGGER.info(f"CV Mean {scoring}: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")
    return cv_scores.tolist()