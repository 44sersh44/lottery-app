"""Сохранение и загрузка обученных моделей с помощью joblib."""

import os
import joblib
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from core.base_predictor import BasePredictor


class ModelManager:
    """Менеджер для сохранения/загрузки моделей с поддержкой двойной лотереи."""
    
    # Версия формата моделей (при изменении логики моделей увеличить)
    MODEL_VERSION = 2
    
    def __init__(self, profile_path: str):
        self.profile_path = profile_path
        self.models_dir = os.path.join(profile_path, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Файл с метаданными моделей
        self.metadata_file = os.path.join(self.models_dir, "metadata.json")
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Загружает метаданные моделей."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Сохраняет метаданные моделей."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Ошибка сохранения метаданных: {e}")

    def _get_model_filename(self, name: str, field: int = 1) -> str:
        """
        Генерирует имя файла для модели.
        
        Args:
            name: Имя модели (например, "Frequency")
            field: Поле (1 или 2)
        
        Returns:
            Имя файла
        """
        if field == 2:
            return f"{name}_field2.pkl"
        return f"{name}.pkl"

    def save_model(self, name: str, predictor: BasePredictor, field: int = 1) -> bool:
        """
        Сохраняет модель в файл.
        
        Args:
            name: Имя модели
            predictor: Обученный предиктор
            field: Поле (1 или 2)
        
        Returns:
            True при успехе, False при ошибке
        """
        filename = self._get_model_filename(name, field)
        filepath = os.path.join(self.models_dir, filename)
        
        # Сохраняем метаданные
        self._metadata[filename] = {
            "name": name,
            "field": field,
            "model_version": self.MODEL_VERSION,
            "saved_at": datetime.now().isoformat(),
            "is_trained": predictor.is_trained,
            "is_double": getattr(predictor, 'is_double', False),
            "total_numbers": getattr(predictor, 'total_numbers', 0),
            "pick_count": getattr(predictor, 'pick_count', 0)
        }
        
        try:
            joblib.dump(predictor, filepath)
            self._save_metadata()
            return True
        except Exception as e:
            print(f"[ERR] Ошибка сохранения модели {name} (поле{field}): {e}")
            # Удаляем метаданные при ошибке
            if filename in self._metadata:
                del self._metadata[filename]
                self._save_metadata()
            return False

    def load_model(self, name: str, field: int = 1, 
                   check_params: bool = True,
                   expected_total: int = None, 
                   expected_pick: int = None) -> Optional[BasePredictor]:
        """
        Загружает модель из файла.
        
        Args:
            name: Имя модели
            field: Поле (1 или 2)
            check_params: Проверять соответствие параметров лотереи
            expected_total: Ожидаемое максимальное число
            expected_pick: Ожидаемое количество чисел
        
        Returns:
            Загруженный предиктор или None при ошибке
        """
        filename = self._get_model_filename(name, field)
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        # Проверяем совместимость версии
        if filename in self._metadata:
            meta = self._metadata[filename]
            if meta.get("model_version", 0) != self.MODEL_VERSION:
                print(f"[WARN] Модель {name} (поле{field}) сохранена с версией {meta.get('model_version', 0)}, "
                      f"текущая версия {self.MODEL_VERSION}. Рекомендуется переобучить.")
            
            # Проверяем параметры лотереи
            if check_params:
                if expected_total is not None:
                    saved_total = meta.get("total_numbers", 0)
                    if saved_total != 0 and saved_total != expected_total:
                        print(f"[WARN] Модель {name} (поле{field}) обучена для чисел 1-{saved_total}, "
                              f"текущий диапазон 1-{expected_total}. Рекомендуется переобучить.")
                
                if expected_pick is not None:
                    saved_pick = meta.get("pick_count", 0)
                    if saved_pick != 0 and saved_pick != expected_pick:
                        print(f"[WARN] Модель {name} (поле{field}) обучена для {saved_pick} чисел, "
                              f"текущее количество {expected_pick}. Рекомендуется переобучить.")
        
        try:
            model = joblib.load(filepath)
            return model
        except Exception as e:
            print(f"[ERR] Ошибка загрузки модели {name} (поле{field}): {e}")
            return None

    def delete_model(self, name: str, field: int = 1) -> bool:
        """
        Удаляет модель.
        
        Args:
            name: Имя модели
            field: Поле (1 или 2)
        
        Returns:
            True при успехе, False при ошибке
        """
        filename = self._get_model_filename(name, field)
        filepath = os.path.join(self.models_dir, filename)
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                if filename in self._metadata:
                    del self._metadata[filename]
                    self._save_metadata()
                return True
            except Exception as e:
                print(f"[ERR] Ошибка удаления модели {name}: {e}")
                return False
        return False

    def list_models(self, field: int = None) -> Union[Dict[int, List[str]], List[str]]:
        """
        Возвращает список сохранённых моделей.
        
        Args:
            field: Фильтр по полю (1, 2 или None для всех)
        
        Returns:
            Если field указан — список имён моделей
            Если field не указан — словарь {1: [...], 2: [...]}
        """
        result = {1: [], 2: []}
        
        for filename in os.listdir(self.models_dir):
            if not filename.endswith('.pkl'):
                continue
            
            if filename.endswith("_field2.pkl"):
                # Имя файла: Markov_field2.pkl -> Markov
                real_name = filename[:-11]  # убираем _field2.pkl
                result[2].append(real_name)
            else:
                # Имя файла: Markov.pkl -> Markov
                name = filename[:-4]  # убираем .pkl
                result[1].append(name)
        
        if field is not None:
            return result.get(field, [])
        return result

    def get_models_by_field(self, field: int = 1) -> List[str]:
        """
        Возвращает список моделей для указанного поля.
        
        Args:
            field: Поле (1 или 2)
        
        Returns:
            Список имён моделей
        """
        models = self.list_models(field=field)
        return models if isinstance(models, list) else []

    def get_model_count(self) -> Dict[int, int]:
        """
        Возвращает количество моделей для каждого поля.
        
        Returns:
            Словарь {1: количество_поле1, 2: количество_поле2}
        """
        models = self.list_models()
        return {1: len(models[1]), 2: len(models[2])}

    def get_all_models(self) -> Dict[int, List[str]]:
        """Возвращает все модели сгруппированные по полям."""
        return self.list_models()

    def get_model_info(self, name: str, field: int = 1) -> Optional[Dict[str, Any]]:
        """Возвращает метаданные модели."""
        filename = self._get_model_filename(name, field)
        return self._metadata.get(filename)

    def model_exists(self, name: str, field: int = 1) -> bool:
        """Проверяет, существует ли модель."""
        filename = self._get_model_filename(name, field)
        filepath = os.path.join(self.models_dir, filename)
        return os.path.exists(filepath)

    def clear_old_models(self, days_old: int = 30) -> int:
        """
        Удаляет модели старше указанного количества дней.
        
        Args:
            days_old: Возраст в днях
        
        Returns:
            Количество удалённых моделей
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days_old)
        removed = 0
        
        for filename, meta in list(self._metadata.items()):
            saved_at = meta.get("saved_at")
            if saved_at:
                try:
                    saved_time = datetime.fromisoformat(saved_at)
                    if saved_time < cutoff:
                        filepath = os.path.join(self.models_dir, filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        del self._metadata[filename]
                        removed += 1
                except Exception:
                    pass
        
        if removed > 0:
            self._save_metadata()
            print(f"[MODEL] Удалено {removed} старых моделей (старше {days_old} дней)")
        
        return removed

    def clear_all_models(self) -> int:
        """Удаляет все модели профиля."""
        removed = 0
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.models_dir, filename)
                try:
                    os.remove(filepath)
                    removed += 1
                except Exception:
                    pass
        
        # Также удаляем файлы ансамблей, если они есть в корне профиля
        ensemble_names = ["Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"]
        for name in ensemble_names:
            for field in [1, 2]:
                filename = self._get_model_filename(name, field)
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        removed += 1
                    except Exception:
                        pass
        
        self._metadata = {}
        self._save_metadata()
        print(f"[MODEL] Удалено {removed} моделей")
        return removed

    def get_model_files(self) -> List[str]:
        """Возвращает список всех .pkl файлов в папке моделей."""
        return [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает все метаданные."""
        return self._metadata.copy()

    def __repr__(self) -> str:
        counts = self.get_model_count()
        return f"<ModelManager field1={counts[1]}, field2={counts[2]}>"