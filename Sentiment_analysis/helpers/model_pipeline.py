"""
Text Classification Modeling Pipeline
-------------------------------------
This module provides utility functions to evaluate text classification models using sklearn pipelines.
"""

# Standard library
import numpy as np  # Numerical operations and array manipulation

# Visualization
import seaborn as sns  # Statistical data visualization (e.g., heatmaps)
import matplotlib.pyplot as plt  # General plotting (e.g., ROC curves)

# Scikit-learn modules
from sklearn.metrics import (
    confusion_matrix,  # Confusion matrix for classification
    classification_report  # Precision, recall, F1-score summary
)


def model_Evaluate(model, X_test, y_test):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)