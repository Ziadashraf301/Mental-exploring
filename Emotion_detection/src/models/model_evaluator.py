import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from utils.pipeline_logger import pipeline_logger


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
    
    metrics_dict = {}
    
    if model_type == 'sklearn':
        pipeline_logger.info("Evaluating sklearn model...")
        
        # Predictions
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # Probabilities for log loss
        prob_train = model.predict_proba(x_train)
        prob_test = model.predict_proba(x_test)

        # Metrics
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        loss_train = log_loss(y_train, prob_train)
        loss_test = log_loss(y_test, prob_test)

        metrics_dict = {
            'acc_train': acc_train,
            'acc_test': acc_test,
            'f1_train': f1_train,
            'f1_test': f1_test,
            'loss_train': loss_train,
            'loss_test': loss_test
        }

        # Print report
        print('\n' + '=' * 50)
        print('Sklearn Model Evaluation Results')
        print('=' * 50)
        print(f'Loss Train  : {loss_train:.4f}')
        print(f'Loss Test   : {loss_test:.4f}')
        print(f'Accuracy Train : {acc_train*100:.2f}%')
        print(f'Accuracy Test  : {acc_test*100:.2f}%')
        print(f'F1 Train : {f1_train*100:.2f}%')
        print(f'F1 Test  : {f1_test*100:.2f}%')
        print('=' * 50)
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred_test, target_names=['Sad', 'Happy']))

        pipeline_logger.info(f"Sklearn model - Test Acc: {acc_test:.3f}, F1: {f1_test:.3f}")

        # Create comprehensive plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Accuracy comparison
        axes[0].bar(['Train', 'Test'], [acc_train*100, acc_test*100], 
                    color=['skyblue', 'salmon'], edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate([acc_train*100, acc_test*100]):
            axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Plot 2: F1 Score comparison
        axes[1].bar(['Train', 'Test'], [f1_train*100, f1_test*100], 
                    color=['lightgreen', 'orange'], edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('F1 Score (%)', fontsize=12)
        axes[1].set_title('Train vs Test F1 Score', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate([f1_train*100, f1_test*100]):
            axes[1].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Plot 3: Loss comparison
        axes[2].bar(['Train', 'Test'], [loss_train, loss_test], 
                    color=['purple', 'red'], edgecolor='black', alpha=0.7)
        axes[2].set_ylabel('Log Loss', fontsize=12)
        axes[2].set_title('Train vs Test Loss', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate([loss_train, loss_test]):
            axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=150, bbox_inches='tight')
            pipeline_logger.info(f"Saved evaluation plot to {save_path}_metrics.png")
        plt.close()

    elif model_type == 'tensorflow':
        pipeline_logger.info("Evaluating TensorFlow model...")
        
        # Evaluate model
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=1)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        
        # Get predictions for classification report
        y_pred_test = (model.predict(x_test, verbose=1) > 0.5).astype(int).flatten()
        
        metrics_dict = {
            'acc_train': train_acc,
            'acc_test': test_acc,
            'loss_train': train_loss,
            'loss_test': test_loss
        }
        
        # Print report
        print('\n' + '=' * 50)
        print('TensorFlow Model Evaluation Results')
        print('=' * 50)
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_acc*100:.2f}%")
        print(f"Test Loss      : {test_loss:.4f}")
        print(f"Test Accuracy  : {test_acc*100:.2f}%")
        print('=' * 50)
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred_test, target_names=['Sad', 'Happy']))
        
        pipeline_logger.info(f"TensorFlow model - Test Acc: {test_acc:.3f}, Loss: {test_loss:.3f}")

        # Create comprehensive plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Accuracy comparison
        axes[0].bar(['Train', 'Test'], [train_acc*100, test_acc*100], 
                    color=['skyblue', 'salmon'], edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate([train_acc*100, test_acc*100]):
            axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Plot 2: Loss comparison
        axes[1].bar(['Train', 'Test'], [train_loss, test_loss], 
                    color=['purple', 'red'], edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Train vs Test Loss', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate([train_loss, test_loss]):
            axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=150, bbox_inches='tight')
            pipeline_logger.info(f"Saved evaluation plot to {save_path}_metrics.png")
        plt.close()
    
    else:
        raise ValueError("model_type must be either 'sklearn' or 'tensorflow'")
    
    return metrics_dict