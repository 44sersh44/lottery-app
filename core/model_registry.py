# core/model_registry.py
import os
import importlib
import inspect
from typing import Dict, Type
from predictors.base import BasePredictor

def discover_models() -> Dict[str, Type]:
    """
    Сканирует папку predictors и регистрирует все классы, наследующие BasePredictor.
    Пропускает ансамбли и вспомогательные классы.
    """
    registry = {}
    predictors_dir = os.path.join(os.path.dirname(__file__), '..', 'predictors')
    if not os.path.exists(predictors_dir):
        return registry

    # Имена, которые нужно пропустить (ансамбли, базовые классы)
    skip_names = {
        "Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble",
        "BlendingEnsemble", "MetaEnsemble", "BasePredictor"
    }

    for filename in os.listdir(predictors_dir):
        if not filename.endswith('.py') or filename.startswith('_'):
            continue
        if filename in ('base.py', '__init__.py'):
            continue
        module_name = filename[:-3]  # убираем .py
        try:
            module = importlib.import_module(f'predictors.{module_name}')
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name in skip_names or not inspect.isclass(obj):
                    continue
                # Проверяем, является ли класс наследником BasePredictor
                if issubclass(obj, BasePredictor) and obj is not BasePredictor:
                    # Убираем суффикс Predictor для ключа
                    key = name.replace('Predictor', '')
                    registry[key] = obj
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {module_name}: {e}")

    return registry

MODEL_REGISTRY = discover_models()

if __name__ == "__main__":
    print(f"Зарегистрировано моделей: {len(MODEL_REGISTRY)}")
    for name in sorted(MODEL_REGISTRY.keys()):
        print(f"  {name}")