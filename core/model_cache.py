# core/model_cache.py
import os
import joblib
from typing import Optional, Any
from datetime import datetime

class ModelCache:
    """Кэширование обученных моделей с использованием joblib."""
    
    def __init__(self, cache_dir: str = "models_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, profile_name: str, model_name: str, field: int = 1) -> str:
        """Возвращает путь к файлу кэша для конкретной модели."""
        # Создаём подпапку для профиля
        profile_dir = os.path.join(self.cache_dir, profile_name)
        os.makedirs(profile_dir, exist_ok=True)
        filename = f"{model_name}_field{field}.pkl"
        return os.path.join(profile_dir, filename)
    
    def save_model(self, profile_name: str, model_name: str, model: Any, field: int = 1, metadata: dict = None):
        """Сохраняет модель в кэш."""
        path = self.get_cache_path(profile_name, model_name, field)
        data = {
            'model': model,
            'saved_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        joblib.dump(data, path)
        return path
    
    def load_model(self, profile_name: str, model_name: str, field: int = 1) -> Optional[Any]:
        """Загружает модель из кэша, если она существует."""
        path = self.get_cache_path(profile_name, model_name, field)
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                return data['model']
            except Exception as e:
                print(f"Ошибка загрузки модели {model_name}: {e}")
                return None
        return None
    
    def is_cached(self, profile_name: str, model_name: str, field: int = 1) -> bool:
        """Проверяет, существует ли кэшированная модель."""
        return os.path.exists(self.get_cache_path(profile_name, model_name, field))
    
    def clear_cache(self, profile_name: Optional[str] = None, model_name: Optional[str] = None):
        """Очищает кэш (все или конкретной модели)."""
        if profile_name:
            profile_dir = os.path.join(self.cache_dir, profile_name)
            if os.path.exists(profile_dir):
                if model_name:
                    # Удаляем конкретную модель
                    for field in [1, 2]:
                        path = self.get_cache_path(profile_name, model_name, field)
                        if os.path.exists(path):
                            os.remove(path)
                else:
                    # Удаляем все модели профиля
                    import shutil
                    shutil.rmtree(profile_dir)
        else:
            # Удаляем весь кэш
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)