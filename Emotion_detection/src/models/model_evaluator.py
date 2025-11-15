from src.logger.train_logger import pipeline_logger
from src.utils.metrics_utils import get_predictions, compute_metrics
from src.utils.plot_utils import plot_metrics
from src.utils.report_utils import print_report

def get_report(model, x_train, y_train, x_test, y_test, model_type='sklearn', save_path=None):
    """
    Generate evaluation report for both sklearn and TensorFlow models, with plots.

    Parameters:
    -----------
    model : sklearn or tf.keras.Model
        Trained model to evaluate.
    x_train, y_train : array-like
        Training data and labels.
    x_test, y_test : array-like
        Test data and labels.
    model_type : str, default='sklearn'
        Type of the model: 'sklearn' or 'tensorflow'.
    save_path : str or None
        Path to save the plot. If None, plot is shown but not saved.

    Returns:
    --------
    metrics_dict : dict
        Dictionary with train/test accuracy, F1, and log loss (if applicable).
    """
    
    pipeline_logger.info(f"Evaluating {model_type} model...")

    # 1) predictions
    y_pred_train, prob_train = get_predictions(model, x_train, model_type)
    y_pred_test, prob_test = get_predictions(model, x_test, model_type)

    # 2) metrics
    train_m = compute_metrics(y_train, y_pred_train, prob_train)
    test_m = compute_metrics(y_test, y_pred_test, prob_test)

    # 3) print
    print_report(model_type, train_m, test_m, y_test, y_pred_test)

    # 4) plot
    plot_metrics(train_m, test_m, save_path)

    # 5) logging
    pipeline_logger.info(
        f"{model_type} model - Test Acc: {test_m['accuracy']:.3f}, "
        f"F1: {test_m['f1']:.3f}, LogLoss: {test_m['logloss']:.3f}"
    )

    return {
        "train": train_m,
        "test": test_m
    }