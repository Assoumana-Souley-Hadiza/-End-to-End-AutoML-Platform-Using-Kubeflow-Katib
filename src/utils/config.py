"""
Configuration management
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._load_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_env_vars(self) -> None:
        """Load environment variables and override config"""
        env_mappings = {
            "KUBECONFIG_PATH": "kubernetes.kubeconfig_path",
            "KUBERNETES_NAMESPACE": "kubernetes.namespace",
            "KFP_HOST": "kubeflow.host",
            "KATIB_NAMESPACE": "katib.namespace",
            "KFSERVING_NAMESPACE": "kfserving.namespace",
            "MODEL_STORAGE_PATH": "model.storage_path",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(config_path, value)
    
    def _set_nested(self, path: str, value: Any) -> None:
        """Set nested config value"""
        keys = path.split(".")
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set config value"""
        self._set_nested(key, value)



