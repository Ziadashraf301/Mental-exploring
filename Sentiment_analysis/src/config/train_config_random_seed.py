import os
import random
import numpy as np
from Sentiment_analysis.src.logger.train_logger import get_logger

def set_random_seeds(config):
    """Set random seeds for reproducibility"""
    
    LOGGER = get_logger()
   
    seeds = config.random_seeds
    
    # Python
    random.seed(seeds['python_seed'])
    
    # NumPy
    np.random.seed(seeds['numpy_seed'])
    
    # Set for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seeds['python_seed'])
    
    LOGGER.info(f"Random seeds set: {seeds}")