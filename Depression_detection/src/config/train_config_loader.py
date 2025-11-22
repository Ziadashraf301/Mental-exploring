"""
Configuration loader for the Depression Detection Pipeline
Loads settings from train_config.yaml file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class that loads and provides access to all depression detection settings"""
    
    def __init__(self, config_path: str = "train_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        required_sections = [
            'data', 'mlflow', 'output', 'logging', 'pipelines',
            'cross_validation', 'random_seeds', 'data_loader', 'model_saving'
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_directories(self):
        for dir_path in [self.models_dir, self.plots_dir]:
            full_path = Path(dir_path).resolve()
            full_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== DATA ====================
    @property
    def raw_data_path(self) -> str:
        return self.config['data']['raw_data_path']
    
    @property
    def processed_data_path(self) -> str:
        return self.config['data']['processed_data_path']
    
    # ==================== MLFLOW ====================
    @property
    def mlflow_experiment_name(self) -> str:
        return self.config['mlflow']['experiment_name']
    
    @property
    def mlflow_tracking_uri(self) -> str:
        return self.config['mlflow']['tracking_uri']
    
    # ==================== OUTPUT ====================
    @property
    def models_dir(self) -> str:
        return self.config['output']['models_dir']
    
    @property
    def plots_dir(self) -> str:
        return self.config['output']['plots_dir']
    
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
    
    # ==================== PIPELINES ====================
    @property
    def pipelines(self) -> Dict[str, Any]:
        return self.config['pipelines']
    
    def get_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        return self.config['pipelines'].get(pipeline_name, {})
    
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
       
    # ==================== RANDOM SEEDS ====================
    @property
    def random_seeds(self) -> Dict[str, int]:
        return self.config['random_seeds']
    
    # ==================== DATA LOADER ====================
    @property
    def data_loader(self) -> Dict[str, Any]:
        return self.config['data_loader']
    
    # ==================== MODEL SAVING ====================
    @property
    def model_saving(self) -> Dict[str, Any]:
        return self.config['model_saving']
    
    # ==================== UTILS ====================
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def print_config(self):
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
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Return the hyperparameters for a model in classical_ml pipeline (SGD or MNB)."""
        classical_ml = self.get_pipeline('classical_ml').get('models', {})
        return classical_ml.get(model_name.lower(), {}).get('params', {})


# Singleton instance
_config_instance = None

def get_train_config(config_path: str = "train_config.yaml") -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def reload_train_config(config_path: str = "train_config.yaml") -> Config:
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    config = get_train_config()
    config.print_config()
