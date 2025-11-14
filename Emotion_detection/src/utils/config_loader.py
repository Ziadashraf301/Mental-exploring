"""
Configuration loader for the Emotion Detection Pipeline
Loads settings from config.yaml file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class that loads and provides access to all pipeline settings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate that all required configuration sections exist"""
        required_sections = [
            'data', 'mlflow', 'output', 'preprocessing',
            'logistic_regression', 'feedforward_nn', 'cnn_architecture',
            'cnn_training', 'data_augmentation', 'evaluation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_directories(self):
        """Create output directories if they don't exist"""
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # ==================== DATA PATHS ====================
    @property
    def train_images_path(self) -> str:
        return self.config['data']['train_images_path']
    
    @property
    def train_labels_path(self) -> str:
        return self.config['data']['train_labels_path']
    
    @property
    def test_images_path(self) -> str:
        return self.config['data']['test_images_path']
    
    @property
    def test_labels_path(self) -> str:
        return self.config['data']['test_labels_path']
    
    # ==================== MLFLOW SETTINGS ====================
    @property
    def mlflow_experiment_name(self) -> str:
        return self.config['mlflow']['experiment_name']
    
    @property
    def mlflow_tracking_uri(self) -> str:
        return self.config['mlflow']['tracking_uri']
    
    # ==================== OUTPUT DIRECTORIES ====================
    @property
    def models_dir(self) -> str:
        return self.config['output']['models_dir']
    
    @property
    def plots_dir(self) -> str:
        return self.config['output']['plots_dir']
    
    @property
    def logs_dir(self) -> str:
        return self.config['output']['logs_dir']
    
    # ==================== PREPROCESSING ====================
    @property
    def image_size(self) -> tuple:
        return tuple(self.config['preprocessing']['image_size'])
    
    @property
    def normalize(self) -> bool:
        return self.config['preprocessing']['normalize']
    
    @property
    def color_mode(self) -> str:
        return self.config['preprocessing']['color_mode']
    
    # ==================== LOGISTIC REGRESSION ====================
    @property
    def lr_enabled(self) -> bool:
        return self.config['logistic_regression']['enabled']
    
    @property
    def lr_params(self) -> Dict[str, Any]:
        params = self.config['logistic_regression'].copy()
        params.pop('enabled', None)
        return params
    
    # ==================== FEEDFORWARD NN ====================
    @property
    def ffn_enabled(self) -> bool:
        return self.config['feedforward_nn']['enabled']
    
    @property
    def ffn_params(self) -> Dict[str, Any]:
        params = self.config['feedforward_nn'].copy()
        params.pop('enabled', None)
        return params
    
    # ==================== CNN ARCHITECTURE ====================
    @property
    def cnn_enabled(self) -> bool:
        return self.config['cnn_architecture']['enabled']
    
    @property
    def cnn_input_shape(self) -> tuple:
        return tuple(self.config['cnn_architecture']['input_shape'])
    
    @property
    def cnn_conv_layers(self) -> list:
        return self.config['cnn_architecture']['conv_layers']
    
    @property
    def cnn_dense_layers(self) -> list:
        return self.config['cnn_architecture']['dense_layers']
    
    @property
    def cnn_output_layer(self) -> Dict[str, Any]:
        return self.config['cnn_architecture']['output_layer']
    
    # ==================== CNN TRAINING ====================
    @property
    def cnn_training_params(self) -> Dict[str, Any]:
        return self.config['cnn_training']
    
    @property
    def cnn_optimizer(self) -> str:
        return self.config['cnn_training']['optimizer']
    
    @property
    def cnn_learning_rate(self) -> float:
        return self.config['cnn_training']['learning_rate']
    
    @property
    def cnn_loss(self) -> str:
        return self.config['cnn_training']['loss']
    
    @property
    def cnn_epochs(self) -> int:
        return self.config['cnn_training']['epochs']
    
    @property
    def cnn_batch_size(self) -> int:
        return self.config['cnn_training']['batch_size']
    
    @property
    def cnn_validation_split(self) -> float:
        return self.config['cnn_training']['validation_split']
    
    @property
    def cnn_early_stopping(self) -> Dict[str, Any]:
        return self.config['cnn_training']['early_stopping']
    
    # ==================== DATA AUGMENTATION ====================
    @property
    def augmentation_enabled(self) -> bool:
        return self.config['data_augmentation']['enabled']
    
    @property
    def augmentation_params(self) -> Dict[str, Any]:
        params = self.config['data_augmentation'].copy()
        params.pop('enabled', None)
        return params
    
    # ==================== CROSS VALIDATION ====================
    @property
    def cv_enabled(self) -> bool:
        return self.config['cross_validation']['enabled']
    
    @property
    def cv_k_folds(self) -> int:
        return self.config['cross_validation']['k_folds']
    
    @property
    def cv_params(self) -> Dict[str, Any]:
        return self.config['cross_validation']
    
    # ==================== EVALUATION ====================
    @property
    def target_names(self) -> list:
        return self.config['evaluation']['target_names']
    
    @property
    def evaluation_params(self) -> Dict[str, Any]:
        return self.config['evaluation']
    
    # ==================== LOGGING ====================
    @property
    def logging_level(self) -> str:
        return self.config['logging']['level']
    
    @property
    def console_output(self) -> bool:
        return self.config['logging']['console_output']
    
    @property
    def file_output(self) -> bool:
        return self.config['logging']['file_output']
    
    # ==================== PERFORMANCE ====================
    @property
    def use_gpu(self) -> bool:
        return self.config['performance']['use_gpu']
    
    @property
    def gpu_memory_fraction(self) -> float:
        return self.config['performance']['gpu_memory_fraction']
    
    # ==================== RANDOM SEEDS ====================
    @property
    def random_seeds(self) -> Dict[str, int]:
        return self.config['random_seeds']
    
    # ==================== MODEL SAVING ====================
    @property
    def model_saving_params(self) -> Dict[str, Any]:
        return self.config['model_saving']
    
    def get(self, key: str, default=None):
        """Get any configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"
    
    def print_config(self):
        """Print all configuration settings"""
        print("=" * 70)
        print("CONFIGURATION SETTINGS")
        print("=" * 70)
        for section, values in self.config.items():
            print(f"\n[{section.upper()}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")
        print("=" * 70)


# Singleton instance
_config_instance = None

def get_config(config_path: str = "../config.yaml") -> Config:
    """
    Get or create configuration instance (Singleton pattern)
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    Config
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: str = "../config.yaml") -> Config:
    """
    Force reload of configuration
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    Config
        New configuration instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    config.print_config()