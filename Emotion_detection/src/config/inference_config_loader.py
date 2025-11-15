"""
Configuration loader for inference pipeline
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class InferenceConfig:
    """Configuration class for inference pipeline"""
    
    def __init__(self, config_path: str = "inference_config.yaml"):
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
        required_sections = ['mlflow', 'face_detection', 'preprocessing', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_directories(self):
        """Create output directories if they don't exist"""
        # Create logs directory
        log_file = self.config['logging'].get('log_file')
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create results directory if needed
        if self.config.get('output', {}).get('save_results'):
            results_dir = self.config['output']['results_dir']
            Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # ==================== MLFLOW SETTINGS ====================
    @property
    def mlflow_tracking_uri(self) -> str:
        return self.config['mlflow']['tracking_uri']
    
    @property
    def model_name(self) -> str:
        return self.config['mlflow']['model_name']
    
    @property
    def model_version(self) -> str:
        return self.config['mlflow'].get('model_version')
    
    @property
    def model_stage(self) -> str:
        return self.config['mlflow'].get('model_stage')
    
    @property
    def model_uri(self) -> str:
        """Generate model URI based on version or stage"""
        if self.model_stage:
            return f"models:/{self.model_name}@{self.model_stage}"
        else:
            return f"models:/{self.model_name}/{self.model_version}"
    
    # ==================== FACE DETECTION ====================
    @property
    def face_confidence_threshold(self) -> float:
        return self.config['face_detection']['confidence_threshold']
    
    # ==================== PREPROCESSING ====================
    @property
    def image_size(self) -> tuple:
        return tuple(self.config['preprocessing']['image_size'])
    
    @property
    def normalize(self) -> bool:
        return self.config['preprocessing']['normalize']
    
    # ==================== LOGGING ====================
    @property
    def log_level(self) -> str:
        return self.config['logging']['level']
    
    @property
    def log_file(self) -> str:
        return self.config['logging']['log_file']
    
    @property
    def console_output(self) -> bool:
        return self.config['logging']['console_output']
    
    # ==================== OUTPUT ====================
    @property
    def save_results(self) -> bool:
        return self.config.get('output', {}).get('save_results', False)
    
    @property
    def results_dir(self) -> str:
        return self.config.get('output', {}).get('results_dir', 'results')
    
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
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("=" * 70)
        print("INFERENCE CONFIGURATION")
        print("=" * 70)
        
        print("\n[MLFLOW]")
        print(f"  Tracking URI: {self.mlflow_tracking_uri}")
        print(f"  Model Name: {self.model_name}")
        if self.model_stage:
            print(f"  Model Stage: {self.model_stage}")
        else:
            print(f"  Model Version: {self.model_version}")
        print(f"  Model URI: {self.model_uri}")
        
        print("\n[FACE DETECTION]")
        print(f"  Confidence Threshold: {self.face_confidence_threshold}")
        
        print("\n[PREPROCESSING]")
        print(f"  Image Size: {self.image_size}")
        print(f"  Normalize: {self.normalize}")
        
        print("\n[LOGGING]")
        print(f"  Level: {self.log_level}")
        print(f"  Log File: {self.log_file}")
        print(f"  Console Output: {self.console_output}")
        
        print("=" * 70)


# Singleton instance
_config_instance = None


def get_inference_config(config_path: str = "config/inference_config.yaml") -> InferenceConfig:
    """
    Get or create inference configuration instance (Singleton pattern)
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    InferenceConfig
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = InferenceConfig(config_path)
    return _config_instance


def reload_inference_config(config_path: str = "config/inference_config.yaml") -> InferenceConfig:
    """
    Force reload of configuration
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    InferenceConfig
        New configuration instance
    """
    global _config_instance
    _config_instance = InferenceConfig(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_inference_config()
    config.print_config()