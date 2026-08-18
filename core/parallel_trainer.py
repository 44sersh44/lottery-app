"""Параллельное обучение моделей с использованием всех ядер CPU."""

import multiprocessing as mp
from typing import List, Dict, Any, Tuple
from functools import partial
import os
import sys


class ParallelTrainer:
    """
    Класс для параллельного обучения моделей.
    Использует multiprocessing для загрузки всех ядер CPU.
    """
    
    def __init__(self, max_workers: int = None):
        """
        Args:
            max_workers: Максимальное количество параллельных процессов.
                         По умолчанию = количество ядер CPU - 1
        """
        cpu_count = mp.cpu_count()
        if max_workers is None:
            self.max_workers = max(1, cpu_count - 1)  # Оставляем одно ядро для системы
        else:
            self.max_workers = max(1, min(max_workers, cpu_count))
        
        print(f"[PARALLEL] Инициализация: {self.max_workers} параллельных процессов (всего ядер: {cpu_count})")
    
    def train_models_parallel(self, models: Dict[str, Any], train_blocks: List[List[int]],
                              model_manager=None, field: int = 1) -> Dict[str, Any]:
        """
        Параллельно обучает список моделей.
        
        Args:
            models: Словарь {имя_модели: объект_модели}
            train_blocks: Данные для обучения
            model_manager: Менеджер моделей для сохранения (опционально)
            field: Поле (1 или 2) для сохранения
        
        Returns:
            Словарь с обученными моделями и статусами
        """
        if not models:
            return {}
        
        # Подготавливаем аргументы для каждого процесса
        args_list = []
        for name, model in models.items():
            # Исключаем ансамбли из параллельного обучения
            if name in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                continue
            args_list.append((name, model, train_blocks, field))
        
        if not args_list:
            return {}
        
        print(f"[PARALLEL] Запуск обучения {len(args_list)} моделей в {self.max_workers} потоков...")
        
        # Используем пул процессов
        with mp.Pool(processes=self.max_workers) as pool:
            # Частичная функция с фиксированными аргументами
            train_func = partial(self._train_single_model, model_manager=model_manager)
            
            # Запускаем параллельное обучение
            results = pool.map(train_func, args_list)
        
        # Собираем результаты
        trained_models = {}
        for name, model, success, error in results:
            if success:
                trained_models[name] = model
            else:
                print(f"  [ERR] Ошибка обучения {name}: {error}")
        
        print(f"[PARALLEL] Обучено: {len(trained_models)}/{len(args_list)} моделей")
        
        return trained_models
    
    @staticmethod
    def _train_single_model(args: Tuple, model_manager=None) -> Tuple[str, Any, bool, str]:
        """
        Обучает одну модель (запускается в отдельном процессе).
        
        Args:
            args: (name, model, train_blocks, field)
            model_manager: Менеджер моделей (передаётся через partial)
        
        Returns:
            (name, model, success, error_message)
        """
        name, model, train_blocks, field = args
        
        try:
            # Обучаем модель
            model.fit(train_blocks)
            
            # Сохраняем модель, если есть менеджер
            if model_manager:
                model_manager.save_model(name, model, field=field)
            
            return (name, model, True, "")
        except Exception as e:
            return (name, None, False, str(e))
    
    def train_models_chunked(self, models: Dict[str, Any], train_blocks: List[List[int]],
                             chunk_size: int = 10, model_manager=None, field: int = 1) -> Dict[str, Any]:
        """
        Обучает модели чанками (порциями) для контроля памяти.
        
        Args:
            models: Словарь {имя_модели: объект_модели}
            train_blocks: Данные для обучения
            chunk_size: Размер чанка (сколько моделей обучать за раз)
            model_manager: Менеджер моделей
            field: Поле (1 или 2)
        
        Returns:
            Словарь с обученными моделями
        """
        model_list = list(models.items())
        all_trained = {}
        
        for i in range(0, len(model_list), chunk_size):
            chunk = dict(model_list[i:i+chunk_size])
            print(f"[PARALLEL] Обработка чанка {i//chunk_size + 1}/{(len(model_list)-1)//chunk_size + 1}")
            
            trained = self.train_models_parallel(chunk, train_blocks, model_manager, field)
            all_trained.update(trained)
        
        return all_trained


class ParallelPredictor:
    """Класс для параллельного получения прогнозов от моделей."""
    
    def __init__(self, max_workers: int = None):
        cpu_count = mp.cpu_count()
        self.max_workers = max_workers or max(1, cpu_count - 1)
    
    def predict_parallel(self, models: Dict[str, Any], n_predictions: int = 1) -> Dict[str, Any]:
        """
        Параллельно получает прогнозы от всех моделей.
        
        Args:
            models: Словарь {имя_модели: объект_модели}
            n_predictions: Количество прогнозов
        
        Returns:
            Словарь {имя_модели: прогноз}
        """
        if not models:
            return {}
        
        args_list = [(name, model, n_predictions) for name, model in models.items()]
        
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(self._predict_single_model, args_list)
        
        predictions = {}
        for name, pred, error in results:
            if pred is not None:
                predictions[name] = pred
            else:
                print(f"  [ERR] Ошибка прогноза {name}: {error}")
        
        return predictions
    
    @staticmethod
    def _predict_single_model(args: Tuple) -> Tuple[str, Any, str]:
        """Получает прогноз от одной модели (в отдельном процессе)."""
        name, model, n_predictions = args
        try:
            pred = model.predict(n_predictions)
            return (name, pred, "")
        except Exception as e:
            return (name, None, str(e))