# core/config.py
import os
import yaml
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Загружает конфигурацию из config.yaml."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            # Конфиг по умолчанию
            self._config = {
                'app': {'profiles_dir': 'profiles', 'models_cache_dir': 'models_cache'},
                'models': {'active_models': []},
                'training': {'test_ratio': 0.2, 'parallel': True, 'max_workers': 4}
            }
    
    def get(self, key: str, default=None):
        """Получает значение по ключу с точкой (например, 'app.profiles_dir')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value):
        """Устанавливает значение (используется для переопределения во время выполнения)."""
        keys = key.split('.')
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

# Глобальный объект конфигурации
config = Config()