"""Управление параметрами методов прогнозирования."""
import os
import json
import yaml
from typing import Any, Dict, Optional, List
from copy import deepcopy


class ParameterManager:
    """
    Управление параметрами методов прогнозирования.
    Поддерживает сохранение/загрузку, валидацию и значения по умолчанию.
    """
    
    # === ВСТРОЕННЫЕ ДЕФОЛТЫ (используются, если нет в config.yaml) ===
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        # Статистические модели
        "MovingAverage": {"window": 5},
        "Bayesian": {"alpha": 0.5},
        "Bootstrap": {"n_resamples": 100},
        "ExpSmoothing": {"alpha": 0.3},
        "GeometricMean": {},
        "HarmonicMean": {},
        "Median": {},
        
        # Классические модели
        "Frequency": {},
        "Positional": {},
        "Poisson": {"k": 1},
        "Markov": {"order": 1},
        "WeightedFreq": {"decay": 0.9},
        "LinearReg": {},
        "ARIMA": {"order": (5, 1, 0)},
        
        # Машинное обучение
        "RandomForest": {"n_estimators": 50},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        
        # Линейные модели
        "Ridge": {"alpha": 1.0},
        "Lasso": {"alpha": 1.0},
        "ElasticNet": {"alpha": 1.0, "l1_ratio": 0.5},
        "Huber": {"epsilon": 1.35, "alpha": 0.0001},
        "TheilSen": {},
        "OMP": {"n_nonzero_coefs": None},
        "BayesianARD": {},
        "PassiveAggressive": {"C": 1.0},
        
        # SVM/KNN
        "SVM": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "KNN": {"n_neighbors": 5},
        "KNNPattern": {"n_neighbors": 3},
        "DBSCAN": {"eps": 0.5, "min_samples": 2},
        "IsolationForest": {"contamination": 0.1},
        
        # Ансамбли sklearn
        "ExtraTrees": {"n_estimators": 100},
        "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.1},
        "AdaBoost": {"n_estimators": 50, "learning_rate": 1.0},
        "Bagging": {"n_estimators": 10},
        "HistGradientBoosting": {"max_iter": 100, "learning_rate": 0.1},
        
        "BayesianProbability": {},
        "MonteCarlo": {"n_simulations": 100000},
        
        # Базовые sklearn
        "GaussianNB": {},
        "BernoulliNB": {"alpha": 1.0},
        "QuantileRegressor": {"quantile": 0.5},
        "RidgeCV": {"alphas": [0.1, 1.0, 10.0]},
        
        # Ансамбли
        "Ensemble": {"model_selection": "top10", "strategy": "vote"},
        "VotingEnsemble": {"voting": "soft", "weights": None},
        "WeightedEnsemble": {"performance_based": True, "min_weight": 0.01, "use_ranking": True},
        "StackingEnsemble": {"use_features": True, "cv_folds": 5, "meta_model": "ridge"},
        "BlendingEnsemble": {"meta_model": "ridge", "n_folds": 5, "use_original_features": False},
        
        "Genetic": {"population_size": 100, "generations": 50, "mutation_rate": 0.1},
        "LSTM": {"lookback": 3, "epochs": 10},
        "Prophet": {"seasonality_mode": "additive", "yearly_seasonality": False},
        
        # Специальные для поля2
        "Field2": {"model": "Frequency"},
    }

    def __init__(self, profile_path: str):
        self.profile_path = profile_path
        self.params_file = os.path.join(profile_path, "model_params.json")
        
        # === ЗАГРУЖАЕМ ДЕФОЛТЫ ИЗ CONFIG.YAML (ЕСЛИ ЕСТЬ) ===
        self.default_params = self._load_defaults_from_config()
        if self.default_params is None:
            self.default_params = self.DEFAULT_PARAMS  # fallback на встроенные
        
        # Загружаем сохранённые параметры пользователя
        self.params: Dict[str, Dict[str, Any]] = self._load()
        # Автоматически заполняем недостающие параметры значениями по умолчанию
        self._ensure_defaults()

    def _load_defaults_from_config(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Загружает дефолтные параметры из config.yaml.
        Ищет в секциях: models.default_params или default_params.
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config:
                    # Сначала проверяем models.default_params (как в вашем config.yaml)
                    if 'models' in config and 'default_params' in config['models']:
                        return config['models']['default_params']
                    # Если нет, проверяем корневой default_params
                    if 'default_params' in config:
                        return config['default_params']
        except Exception as e:
            print(f"[WARN] Ошибка загрузки default_params из config.yaml: {e}")
        return None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Загружает параметры из файла с обработкой ошибок."""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Преобразуем списки обратно в кортежи для order
                    for model_name, params in loaded.items():
                        if "order" in params and isinstance(params["order"], list):
                            params["order"] = tuple(params["order"])
                        # Преобразуем списки обратно в кортежи для alphas (если нужно)
                        if "alphas" in params and isinstance(params["alphas"], list):
                            # alphas могут оставаться списками
                            pass
                    print(f"[PARAMS] Загружены параметры для {len(loaded)} моделей")
                    return loaded
            except json.JSONDecodeError as e:
                print(f"[ERR] Ошибка чтения {self.params_file}: {e}")
                print(f"[WARN] Создаётся новый файл параметров")
                return {}
            except Exception as e:
                print(f"[ERR] Неожиданная ошибка: {e}")
                return {}
        return {}

    def _save(self) -> None:
        """Сохраняет параметры в файл (атомарно через временный файл)."""
        try:
            os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
            temp_file = self.params_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.params_file):
                os.replace(temp_file, self.params_file)
            else:
                os.rename(temp_file, self.params_file)
        except Exception as e:
            print(f"[ERR] Ошибка сохранения параметров: {e}")

    def _ensure_defaults(self) -> None:
        """Заполняет отсутствующие параметры значениями по умолчанию."""
        modified = False
        for model_name, default_params in self.default_params.items():
            if model_name not in self.params:
                self.params[model_name] = deepcopy(default_params)
                modified = True
            else:
                for param_name, default_value in default_params.items():
                    if param_name not in self.params[model_name]:
                        self.params[model_name][param_name] = deepcopy(default_value)
                        modified = True
        if modified:
            self._save()

    def get_param(self, model_name: str, param_name: str, default: Any = None) -> Any:
        """Получает значение параметра модели."""
        if model_name in self.params and param_name in self.params[model_name]:
            return self.params[model_name][param_name]
        if model_name in self.default_params and param_name in self.default_params[model_name]:
            return self.default_params[model_name][param_name]
        return default

    def set_param(self, model_name: str, param_name: str, value: Any) -> None:
        """Устанавливает значение параметра модели."""
        if value is None:
            print(f"[WARN] {model_name}.{param_name} = None — пропущено")
            return
        if model_name not in self.params:
            self.params[model_name] = {}
        validated = self._validate_param(model_name, param_name, value)
        self.params[model_name][param_name] = validated
        self._save()
        print(f"[PARAMS] {model_name}.{param_name} = {validated}")

    def _validate_param(self, model_name: str, param_name: str, value: Any) -> Any:
        """Валидирует значение параметра."""
        # Валидация для конкретных параметров
        if param_name == "window":
            if not isinstance(value, int) or value < 1:
                print(f"[WARN] window должен быть >= 1, получено {value}. Используется 5")
                return 5
            return value
        elif param_name == "alpha" and model_name != "RidgeCV":
            try:
                val = float(value)
                if val < 0 or val > 1:
                    print(f"[WARN] alpha должен быть между 0 и 1, получено {value}. Используется 0.5")
                    return 0.5
                return val
            except:
                return 0.5
        elif param_name in ("n_estimators", "n_resamples", "max_iter"):
            if not isinstance(value, int) or value < 1:
                print(f"[WARN] {param_name} должен быть >= 1, получено {value}. Используется 100")
                return 100
            return value
        elif param_name == "order":
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                print(f"[WARN] order должен быть кортежем (p,d,q), получено {value}. Используется (5,1,0)")
                return (5, 1, 0)
            return value
        elif param_name == "decay":
            try:
                val = float(value)
                if val < 0 or val > 1:
                    print(f"[WARN] decay должен быть между 0 и 1, получено {value}. Используется 0.9")
                    return 0.9
                return val
            except:
                return 0.9
        elif param_name == "learning_rate":
            try:
                val = float(value)
                if val <= 0 or val > 1:
                    print(f"[WARN] learning_rate должен быть >0 и <=1, получено {value}. Используется 0.1")
                    return 0.1
                return val
            except:
                return 0.1
        elif param_name == "max_depth":
            if not isinstance(value, int) or value < 1 or value > 20:
                print(f"[WARN] max_depth должен быть 1-20, получено {value}. Используется 3")
                return 3
            return value
        elif param_name == "n_nonzero_coefs":
            if value is not None and (not isinstance(value, int) or value < 1):
                print(f"[WARN] n_nonzero_coefs должен быть >= 1 или None, получено {value}")
                return None
            return value
        return value

    def get_all_params(self, model_name: str) -> Dict[str, Any]:
        """Возвращает все параметры модели."""
        result = deepcopy(self.default_params.get(model_name, {}))
        if model_name in self.params:
            result.update(self.params[model_name])
        return result

    def reset_param(self, model_name: str, param_name: str) -> bool:
        """Сбрасывает параметр к значению по умолчанию."""
        if model_name in self.default_params and param_name in self.default_params[model_name]:
            self.set_param(model_name, param_name, self.default_params[model_name][param_name])
            return True
        if model_name in self.params and param_name in self.params[model_name]:
            del self.params[model_name][param_name]
            self._save()
            return True
        return False

    def reset_model(self, model_name: str) -> bool:
        """Сбрасывает все параметры модели к значениям по умолчанию."""
        if model_name in self.default_params:
            self.params[model_name] = deepcopy(self.default_params[model_name])
            self._save()
            return True
        return False

    def reset_all(self) -> None:
        """Сбрасывает все параметры всех моделей к значениям по умолчанию."""
        self.params = deepcopy(self.default_params)
        self._save()
        print("[PARAMS] Все параметры сброшены к значениям по умолчанию")

    def list_models(self) -> List[str]:
        """Возвращает список моделей, у которых есть параметры."""
        models = set(self.params.keys()) | set(self.default_params.keys())
        return sorted(models)

    def export_params(self, filepath: str) -> bool:
        """Экспортирует параметры в JSON файл."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=2, ensure_ascii=False)
            print(f"[PARAMS] Параметры экспортированы в {filepath}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка экспорта: {e}")
            return False

    def import_params(self, filepath: str, merge: bool = True) -> bool:
        """Импортирует параметры из JSON файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                imported = json.load(f)
            if merge:
                for model_name, params in imported.items():
                    if model_name not in self.params:
                        self.params[model_name] = {}
                    self.params[model_name].update(params)
            else:
                self.params = imported
            self._save()
            print(f"[PARAMS] Параметры импортированы из {filepath}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка импорта: {e}")
            return False

    def clean_none_values(self) -> int:
        """Очищает все None значения из параметров."""
        removed_count = 0
        modified = False
        for model_name in list(self.params.keys()):
            model_params = self.params[model_name]
            none_params = [k for k, v in model_params.items() if v is None]
            for param_name in none_params:
                print(f"  [CLEAN] Удалён {model_name}.{param_name} = None")
                del model_params[param_name]
                removed_count += 1
                modified = True
            if not model_params:
                print(f"  [CLEAN] Удалена пустая модель: {model_name}")
                del self.params[model_name]
                modified = True
        if modified:
            self._save()
            print(f"[PARAMS] Очищено {removed_count} параметров с None")
        return removed_count

    def __repr__(self) -> str:
        return f"<ParameterManager models={len(self.params)}>"