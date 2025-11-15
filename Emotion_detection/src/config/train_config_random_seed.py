import os
import random
import numpy as np
import tensorflow as tf
from src.logger.train_logger import pipeline_logger

def set_random_seeds(config):
    """Set random seeds for reproducibility"""
    seeds = config.random_seeds
    
    # Python
    random.seed(seeds['python_seed'])
    
    # NumPy
    np.random.seed(seeds['numpy_seed'])
    
    # TensorFlow
    tf.random.set_seed(seeds['tensorflow_seed'])
    
    # Set for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seeds['python_seed'])
    
    pipeline_logger.info(f"Random seeds set: {seeds}")