import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.pipeline_logger import pipeline_logger
from utils.config_loader import get_config

# --------------------------- SKLEARN MODELS ---------------------------

def train_logistic_regression(X, y):
    """
    Train Logistic Regression using parameters from config.
    """
    config = get_config()
    params = config.lr_params
    
    model = LogisticRegression(
        max_iter=params.get('max_iter', 500),
        penalty=params.get('penalty', 'l2'),
        solver=params.get('solver', 'lbfgs'),
        C=params.get('C', 1.0),
        random_state=params.get('random_state', 42),
        n_jobs=params.get('n_jobs', -1)
    )
    model.fit(X, y)
    
    pipeline_logger.info(f"Trained Logistic Regression with params: {params}")
    return model


def train_ffn(X, y):
    """
    Train Feedforward Neural Network using parameters from config.
    """
    config = get_config()
    params = config.ffn_params
    
    model = MLPClassifier(
        hidden_layer_sizes=tuple(params.get('hidden_layer_sizes', [64, 32, 16])),
        activation=params.get('activation', 'relu'),
        solver=params.get('solver', 'adam'),
        alpha=params.get('alpha', 0.0001),
        batch_size=params.get('batch_size', 32),
        learning_rate=params.get('learning_rate', 'adaptive'),
        learning_rate_init=params.get('learning_rate_init', 0.001),
        max_iter=params.get('max_iter', 100),
        early_stopping=params.get('early_stopping', True),
        validation_fraction=params.get('validation_fraction', 0.1),
        random_state=params.get('random_state', 42),
        verbose=2
    )
    model.fit(X, y)
    
    pipeline_logger.info(f"Trained Feedforward NN with params: {params}")
    return model


# --------------------------- CNN MODELS ---------------------------

def build_cnn():
    """
    Build CNN model using architecture from config.
    """
    config = get_config()
    
    input_shape = config.cnn_input_shape
    conv_layers = config.cnn_conv_layers
    dense_layers = config.cnn_dense_layers
    output_layer = config.cnn_output_layer
    
    model = tf.keras.Sequential()
    
    # Add convolutional layers
    for i, conv_config in enumerate(conv_layers):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(
                filters=conv_config['filters'],
                kernel_size=tuple(conv_config['kernel_size']),
                activation=conv_config['activation'],
                input_shape=input_shape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(
                filters=conv_config['filters'],
                kernel_size=tuple(conv_config['kernel_size']),
                activation=conv_config['activation']
            ))
        
        if 'pool_size' in conv_config:
            model.add(tf.keras.layers.MaxPooling2D(
                pool_size=tuple(conv_config['pool_size'])
            ))
    
    # Flatten
    model.add(tf.keras.layers.Flatten())
    
    # Add dense layers
    for dense_config in dense_layers:
        model.add(tf.keras.layers.Dense(
            units=dense_config['units'],
            activation=dense_config['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(dense_config.get('l2_reg', 0.0))
        ))
        
        if dense_config.get('dropout', 0) > 0:
            model.add(tf.keras.layers.Dropout(dense_config['dropout']))
    
    # Add output layer
    model.add(tf.keras.layers.Dense(
        units=output_layer['units'],
        activation=output_layer['activation']
    ))
    
    # Compile model
    training_config = config.cnn_training_params
    
    # Get optimizer
    optimizer_name = training_config['optimizer']
    learning_rate = training_config['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = 'adam'
    
    model.compile(
        optimizer=optimizer,
        loss=training_config['loss'],
        metrics=['accuracy']
    )
    
    pipeline_logger.info(f"Built CNN model with architecture from config")
    return model


def train_cnn(model, X, y):
    """
    Train CNN using parameters from config.
    """
    config = get_config()
    
    training_params = config.cnn_training_params
    augmentation_params = config.augmentation_params
    cv_params = config.cv_params
    
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    validation_split = training_params['validation_split']
    
    pipeline_logger.info(f"Starting CNN training (epochs={epochs}, batch_size={batch_size})")
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    if training_params['early_stopping']['enabled']:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=training_params['early_stopping']['monitor'],
            patience=training_params['early_stopping']['patience'],
            restore_best_weights=training_params['early_stopping']['restore_best_weights']
        )
        callbacks.append(early_stopping)
    
    # Model checkpoint
    if training_params['checkpoint']['enabled']:
        checkpoint_path = f"{config.models_dir}/cnn_checkpoint.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=training_params['checkpoint']['monitor'],
            save_best_only=training_params['checkpoint']['save_best_only'],
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Data augmentation
    if config.augmentation_enabled:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_params)
        pipeline_logger.info("Data augmentation enabled")
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        pipeline_logger.info("No data augmentation")
    
    # K-Fold cross-validation
    if cv_params['enabled']:
        k_folds = cv_params['k_folds']
        kf = KFold(
            n_splits=k_folds,
            shuffle=cv_params['shuffle'],
            random_state=cv_params['random_state']
        )
        fold_train_scores = []
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            pipeline_logger.info(f"Training fold {fold + 1}/{k_folds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            fold_train_scores.append(train_acc)
            fold_val_scores.append(val_acc)
            pipeline_logger.info(f"Fold {fold + 1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        pipeline_logger.info(f"K-Fold complete. Mean Val Acc: {np.mean(fold_val_scores):.4f}")
        return model, (fold_train_scores, fold_val_scores)
    
    else:
        # Simple fit on all train data 
        history = model.fit(
            datagen.flow(X, y, batch_size=batch_size),
            epochs=epochs,
            verbose=1
        )
        
        pipeline_logger.info(f"Training complete. Final Val Acc: {history.history['accuracy'][-1]:.4f}")
        return model, history