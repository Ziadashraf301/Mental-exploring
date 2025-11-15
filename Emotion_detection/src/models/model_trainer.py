from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from src.logger.train_logger import pipeline_logger
from src.config.train_config_loader import get_config

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


def train_cnn(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_size=32,
    epochs=10,
    augmentation_params=None
):
    """
    Train a CNN model with optional validation and data augmentation.

    Parameters:
    -----------
    model : tf.keras.Model
        CNN model to train.

    X_train : np.ndarray
        Training images.

    y_train : np.ndarray
        Training labels.

    X_val : np.ndarray or None
        Validation images (only used if provided).

    y_val : np.ndarray or None
        Validation labels (only used if provided).

    batch_size : int
        Mini-batch size.

    epochs : int
        Number of training epochs.

    augmentation_params : dict or None
        Parameters passed to ImageDataGenerator.

    Returns:
    --------
    model : tf.keras.Model
        The trained model.

    history : tf.keras.callbacks.History
        The training history object.
    """
    import tensorflow as tf
    from Emotion_detection.src.logger.train_logger import pipeline_logger
    from Emotion_detection.src.config.train_config_loader import get_config

    config = get_config()
    training_params = config.cnn_training_params

    # ===============================
    # 1. Data augmentation
    # ===============================
    if augmentation_params:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_params)
        pipeline_logger.info("Image augmentation enabled.")
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        pipeline_logger.info("No image augmentation applied.")

    # ===============================
    # 2. Callbacks
    # ===============================
    callbacks = []

    # Early Stopping
    es_cfg = training_params.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 5),
                restore_best_weights=es_cfg.get("restore_best_weights", True)
            )
        )

    # Model Checkpoint
    cp_cfg = training_params.get("checkpoint", {})
    if cp_cfg.get("enabled", False):
        checkpoint_path = f"{config.models_dir}/cnn_checkpoint.keras"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=cp_cfg.get("monitor", "val_loss"),
                save_best_only=cp_cfg.get("save_best_only", True),
                verbose=1
            )
        )

    # ===============================
    # 3. Fit model (with or without validation)
    # ===============================
    if X_val is not None and y_val is not None:
        pipeline_logger.info("Training with validation data...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        pipeline_logger.info("Training WITHOUT validation data...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    # ===============================
    # 4. Logging
    # ===============================
    final_acc = history.history["accuracy"][-1]
    pipeline_logger.info(f"Training complete — Final Train Accuracy: {final_acc:.4f}")

    if X_val is not None:
        if "val_accuracy" in history.history:
            pipeline_logger.info(
                f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}"
            )

    return model, history
