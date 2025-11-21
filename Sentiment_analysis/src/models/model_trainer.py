from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from Sentiment_analysis.src.logger.train_logger import get_logger
from Sentiment_analysis.src.config.train_config_loader import get_train_config

LOGGER = get_logger()

# MULTINOMIAL NAIVE BAYES
def train_multinomial_nb(X, y):
    """
    Train Multinomial Naive Bayes using config parameters.
    """
    config = get_train_config()
    params = config.mnb_params

    model = MultinomialNB(
        alpha=params.get("alpha", 1.0),
        fit_prior=params.get("fit_prior", False),
        force_alpha=params.get("force_alpha", True)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained MultinomialNB with params: {params}")

    return model


# LINEAR SVC
def train_linear_svc(X, y):
    """
    Train Linear SVC using config parameters.
    """
    config = get_train_config()
    params = config.svc_params

    model = LinearSVC(
        C=params.get("C", 1.0),
        loss=params.get("loss", "squared_hinge"),
        fit_intercept=params.get("fit_intercept", False),
        tol=params.get("tol", 0.0001),
        max_iter=params.get("max_iter", 1000),
        dual=params.get("dual", True),
        random_state=params.get("random_state", 42)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained Linear SVC with params: {params}")

    return model


# LOGISTIC REGRESSION
def train_logistic_regression(X, y):
    """
    Train Logistic Regression using config parameters.
    """
    config = get_train_config()
    params = config.lr_params

    model = LogisticRegression(
        max_iter=params.get('max_iter', 300),
        penalty=params.get('penalty', 'l2'),
        solver=params.get('solver', 'lbfgs'),
        C=params.get('C', 1.0),
        fit_intercept=params.get('fit_intercept', False),
        random_state=params.get('random_state', 42),
        n_jobs=params.get('n_jobs', -1)
    )
    
    model.fit(X, y)
    LOGGER.info(f"Trained Logistic Regression with params: {params}")

    return model


def run_cv(model, X, y, scoring: str):
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