"""
Text Classification Modeling Pipeline
-------------------------------------
This module provides utility functions to train, evaluate, and analyze
text classification models using sklearn pipelines.
"""

# Standard library
import time  # Measure training and evaluation runtime
import numpy as np  # Numerical operations and array manipulation

# Visualization
import seaborn as sns  # Statistical data visualization (e.g., heatmaps)
import matplotlib.pyplot as plt  # General plotting (e.g., ROC curves)

# Scikit-learn modules
from sklearn.pipeline import Pipeline  # Build ML pipelines
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # Text feature extraction
from sklearn.metrics import (
    accuracy_score,  # Evaluate model accuracy
    confusion_matrix,  # Confusion matrix for classification
    classification_report,  # Precision, recall, F1-score summary
    roc_auc_score,  # AUC metric
    roc_curve  # ROC curve computation
)
from sklearn.model_selection import cross_val_score  # Cross-validation scores


def pipeline_model(model, x_train, y_train):
    """
    Transform the text and train the model using a sklearn Pipeline.

    The pipeline includes:
    1. CountVectorizer to convert text to token counts
    2. TfidfTransformer to convert counts to TF-IDF features
    3. The specified ML model

    Parameters
    ----------
    model : sklearn estimator
        The ML model to train (e.g., LogisticRegression())
    x_train : array-like
        Training input data (text features)
    y_train : array-like
        Training target labels

    Returns
    -------
    pip_model : sklearn.pipeline.Pipeline
        Fitted sklearn pipeline
    model_time : float
        Training time in seconds
    """
    t0 = time.time()

    pip_model = Pipeline([
        ('vect', CountVectorizer(max_features=30000)),
        ('tfidf', TfidfTransformer()),
        ('model', model)
    ])

    pip_model.fit(x_train, y_train)
    model_time = time.time() - t0
    print("Training time (s): {:.5f}".format(model_time))

    return pip_model, model_time


def model_test_results(pip_model, x_test, y_test):
    """
    Evaluate the trained model on test data.

    Metrics computed:
    - Accuracy
    - Confusion matrix
    - Sensitivity (recall for positive class)
    - Specificity (recall for negative class)
    - Classification report
    - Heatmap of normalized confusion matrix

    Parameters
    ----------
    pip_model : sklearn.pipeline.Pipeline
        Trained sklearn pipeline
    x_test : array-like
        Test input data (text features)
    y_test : array-like
        Test target labels
    """
    predictions = pip_model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print('Accuracy score for test data:', score)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print(classification_report(y_test, predictions))

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    sns.heatmap(cm / np.sum(cm), fmt='.2%', cmap="BuPu", annot=True)
    plt.show()


def cross_validation_score(pip_model, x_train, y_train, cv=5):
    """
    Evaluate the model using k-fold cross-validation on the training set.

    Parameters
    ----------
    pip_model : sklearn.pipeline.Pipeline
        Trained sklearn pipeline
    x_train : array-like
        Training input data
    y_train : array-like
        Training target labels
    cv : int, optional (default=5)
        Number of folds for cross-validation
    """
    scores = cross_val_score(pip_model, x_train, y_train, cv=cv)
    print("Mean cross-validation score:", scores.mean())


def auc_roc(pip_model, x_test, y_test, model_name="Model"):
    """
    Plot ROC curve and compute AUC score for the model.

    Parameters
    ----------
    pip_model : sklearn.pipeline.Pipeline
        Trained sklearn pipeline
    x_test : array-like
        Test input data
    y_test : array-like
        Test target labels
    model_name : str, optional
        Name to display on ROC plot (default="Model")
    """
    prob_pred = pip_model.predict_proba(x_test)
    auc_score = roc_auc_score(y_test, prob_pred[:, 1])
    print(f'ROC AUC score for the model: {auc_score:.4f}')

    fpr, tpr, thresh = {}, {}, {}
    for i in range(2):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, prob_pred[:, i], pos_label=i)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr[0], tpr[0], linestyle='--', color='blue', label='Class 0')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1')
    plt.plot([0, 1], [0, 1], ls="--", color='y')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()


def model_train_results(pip_model, x_train, y_train):
    """
    Evaluate the trained model on the training set.

    Metrics computed:
    - Accuracy

    Parameters
    ----------
    pip_model : sklearn.pipeline.Pipeline
        Trained sklearn pipeline
    x_train : array-like
        Training input data
    y_train : array-like
        Training target labels
    """
    predictions = pip_model.predict(x_train)
    score = accuracy_score(y_train, predictions)
    print('Accuracy score for train data:', score)
