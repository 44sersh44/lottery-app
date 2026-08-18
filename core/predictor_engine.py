"""Расширенный движок прогнозирования с полной поддержкой всех моделей для поля2."""

import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from joblib import Parallel, delayed

from core.lottery_data import LotteryData
from core.parameter_manager import ParameterManager
from core.model_cache import ModelCache
from core.model_registry import MODEL_REGISTRY          # <--- ЕДИНСТВЕННЫЙ ИМПОРТ МОДЕЛЕЙ
from core.base_predictor import BasePredictor
from core.model_manager import ModelManager
from core.parallel_trainer import ParallelTrainer, ParallelPredictor


# Подавление предупреждений TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class PredictorEngine:
    def __init__(self, data, profile_path: str):
        self.data = data
        self.profile_path = profile_path
        self.profile_name = os.path.basename(profile_path)
        self.model_cache = ModelCache()  # <-- новый атрибут
        # ... остальная инициализация
        self.data = data
        self.model_manager = ModelManager(profile_path) if profile_path else None
        self.param_manager = ParameterManager(profile_path) if profile_path else None
        self.predictors1: Dict[str, BasePredictor] = {}
        self.predictors2: Dict[str, BasePredictor] = {}   # новые модели для поля2
        self._init_predictors()
        self.profile_path = profile_path

    def load_model_from_cache(self, model_name: str, field: int = 1):
        """Загружает модель из кэша, если она есть и данные не изменились."""
        # Можно добавить проверку хеша данных для инвалидации кэша
        if self.model_cache.is_cached(self.profile_name, model_name, field):
            model = self.model_cache.load_model(self.profile_name, model_name, field)
            if model:
                print(f"[CACHE] Загружена модель {model_name} (поле {field})")
                return model
        return None
    
    def save_model_to_cache(self, model_name: str, model: Any, field: int = 1):
        """Сохраняет модель в кэш."""
        self.model_cache.save_model(self.profile_name, model_name, model, field)
        print(f"[CACHE] Сохранена модель {model_name} (поле {field})")

    def _create_model_instance(self, model_name: str, total: int, pick: int,
                               is_second_field: bool = False) -> BasePredictor:
        """Создаёт модель по имени, используя глобальный реестр."""
        # Получаем класс из реестра
        model_class = MODEL_REGISTRY.get(model_name)
        if model_class is None:
            from predictors.frequency import FrequencyPredictor
            model_class = FrequencyPredictor

        # Получаем параметры из менеджера
        params = {}
        if self.param_manager:
            params = self.param_manager.get_all_params(model_name)

        clean_params = {k: v for k, v in params.items() if v is not None}

        # Специальные корректировки
        if model_name == "Markov":
            order = clean_params.get("order", 1)
            if isinstance(order, (list, tuple)):
                order = order[0] if order else 1
            clean_params["order"] = order
        elif model_name == "ARIMA":
            order = clean_params.get("order", (5, 1, 0))
            if isinstance(order, list):
                order = tuple(order)
            clean_params["order"] = order
        elif model_name == "QuantileRegressor":
            q = clean_params.get("quantile", 0.5)
            if q is None or q <= 0.0 or q >= 1.0:
                q = 0.5
            clean_params["quantile"] = q
        elif model_name == "LSTM":
            if "lookback" not in clean_params:
                clean_params["lookback"] = 5
            if "epochs" not in clean_params:
                clean_params["epochs"] = 20

        try:
            import inspect
            sig = inspect.signature(model_class.__init__)
            init_params = {}
            if 'total_numbers' in sig.parameters:
                init_params['total_numbers'] = total
            if 'pick_count' in sig.parameters:
                init_params['pick_count'] = pick
            if is_second_field:
                init_params['is_second_field'] = True
            all_params = {**init_params, **clean_params}
            model = model_class(**all_params)
        except Exception as e:
            from predictors.frequency import FrequencyPredictor
            model = FrequencyPredictor(total, pick)

        if is_second_field:
            if hasattr(model, 'is_double'):
                model.is_double = False
            if hasattr(model, 'total_numbers2'):
                model.total_numbers2 = 0
            if hasattr(model, 'pick_count2'):
                model.pick_count2 = 0

        return model


    def _init_predictors(self):
        if self.data.is_double():
            total1 = self.data.total_numbers
            pick1 = self.data.pick_count
            total2 = self.data.total_numbers2
            pick2 = self.data.pick_count2
        else:
            total1 = self.data.total_numbers
            pick1 = self.data.pick_count
            total2 = 0
            pick2 = 0

        model_names = list(MODEL_REGISTRY.keys())
        # Исключаем ансамбли, они создаются отдельно
        ensemble_names = {"Ensemble", "VotingEnsemble", "WeightedEnsemble",
                          "StackingEnsemble", "BlendingEnsemble", "MetaEnsemble"}
        model_names = [name for name in model_names if name not in ensemble_names]

        print("[INIT] Инициализация моделей для поля1...")
        for name in model_names:
            try:
                pred = self._create_model_instance(name, total1, pick1, is_second_field=False)
                self.predictors1[name] = pred
            except Exception as e:
                print(f"  [ERR] Ошибка создания модели {name} для поля1: {e}")

        if self.data.is_double():
            print("[INIT] Инициализация моделей для поля2...")
            for name in model_names:
                try:
                    pred = self._create_model_instance(name, total2, pick2, is_second_field=True)
                    self.predictors2[name] = pred
                except Exception as e:
                    print(f"  [ERR] Ошибка создания модели {name} для поля2: {e}")
            print(f"[INIT] Создано {len(self.predictors2)} моделей для поля2")

        print(f"[INIT] Создано {len(self.predictors1)} моделей для поля1")
        
    # Не загружаем ансамбли и не создаём MetaEnsemble
    def _create_meta_ensemble(self):
        """Создаёт MetaEnsemble из лучших моделей ПОСЛЕ инициализации."""
        try:
            from predictors.meta_ensemble import MetaEnsemblePredictor
            
            # Берём лучшие модели для мета-ансамбля
            best_model_names = ["BayesianARD", "GeometricMean", "Positional", "WeightedFreq", "LinearReg"]
            
            best_models = {}
            for name in best_model_names:
                if name in self.predictors1:
                    best_models[name] = self.predictors1[name]
            
            if best_models:
                meta = MetaEnsemblePredictor(best_models, self.data.total_numbers, self.data.pick_count)
                meta.is_trained = True
                self.predictors1["MetaEnsemble"] = meta
                print("[INIT] MetaEnsemble создан для поля1")
                
                # Для поля2
                if self.data.is_double():
                    best_models2 = {}
                    for name in best_model_names:
                        if name in self.predictors2:
                            best_models2[name] = self.predictors2[name]
                    if best_models2:
                        meta2 = MetaEnsemblePredictor(best_models2, self.data.total_numbers2, self.data.pick_count2)
                        meta2.is_trained = True
                        self.predictors2["MetaEnsemble"] = meta2
                        print("[INIT] MetaEnsemble создан для поля2")
        except Exception as e:
            print(f"[WARN] MetaEnsemble не создан: {e}")
        
    def check_predictors_health(self):
        """Проверяет, что все модели корректно инициализированы."""
        print("\n[HEALTH] Проверка состояния предикторов:")
        
        print(f"  Поле1: {len(self.predictors1)} моделей")
        for name, pred in list(self.predictors1.items())[:5]:
            print(f"    {name}: total={pred.total_numbers}, pick={pred.pick_count}, double={getattr(pred, 'is_double', False)}")
        
        if self.data.is_double():
            print(f"  Поле2: {len(self.predictors2)} моделей")
            for name, pred in list(self.predictors2.items())[:5]:
                print(f"    {name}: total={pred.total_numbers}, pick={pred.pick_count}, double={getattr(pred, 'is_double', False)}")
        
        # Проверяем проблемные модели
        problematic = []
        for name, pred in self.predictors2.items():
            if hasattr(pred, 'is_double') and pred.is_double:
                problematic.append(name)
        
        if problematic:
            print(f"\n  [WARN] Модели с is_double=True для поля2: {problematic}")
            print(f"  Это может вызывать проблемы! Рекомендуется исправить.")
        
    def _check_training_data(self, blocks: List[List[int]], min_blocks: int = 3) -> bool:
        """Проверяет, достаточно ли данных для обучения."""
        if len(blocks) < min_blocks:
            print(f"  [WARN] Недостаточно данных: {len(blocks)} блоков, нужно минимум {min_blocks}")
            return False
        
        # Проверяем, что все блоки имеют правильную длину
        expected_len = self.data.pick_count
        for i, block in enumerate(blocks[:5]):  # проверяем первые 5 блоков
            if len(block) != expected_len:
                print(f"  [WARN] Блок {i} имеет длину {len(block)}, ожидается {expected_len}")
                return False
        
        return True

    def train_all(self, train_data: LotteryData = None, force_retrain: bool = False,
                  parallel: bool = False, max_workers: int = None,
                  model_names: List[str] = None) -> None:
        if train_data is None:
            train_data = self.data

        if train_data.is_double():
            train_data.fix_sync()

        blocks1_for_train = list(reversed(train_data.blocks))
        blocks2_for_train = list(reversed(train_data.blocks2)) if train_data.is_double() else []

        trained_count = 0
        loaded_count = 0
        errors_count = 0

        print("\n[TRAIN] Начало обучения моделей...")
        print("=" * 50)

        # Получаем список моделей для обучения
        if model_names is None:
            model_names = list(self.predictors1.keys())
        else:
            model_names = [name for name in model_names if name in self.predictors1]

        # Исключаем ансамбли (они создаются отдельно)
        ensemble_names = {"Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble", "MetaEnsemble", "BlendingEnsemble"}
        model_names = [name for name in model_names if name not in ensemble_names]

        # Исключаем модели поля2 (если они случайно попали в predictors1) - их быть не должно, но на всякий случай
        model_names = [name for name in model_names if not name.endswith("_field2")]

        print(f"[INFO] Будет обучено {len(model_names)} моделей")

        if not model_names:
            print("[TRAIN] Нет моделей для обучения")
            print("=" * 50)
            return

        total_tasks = len(model_names)

        # --- ОБУЧЕНИЕ ПОЛЯ 1 ---
        for name in model_names:
            if name not in self.predictors1:
                print(f"[WARN] Модель {name} не найдена в predictors1")
                continue

            if not force_retrain:
                cached = self.model_cache.load_model(self.profile_name, name, 1)
                if cached is not None:
                    print(f"[CACHE] Загружена модель {name} (поле 1)")
                    self.predictors1[name] = cached
                    loaded_count += 1
                    continue

            try:
                print(f"[TRAIN] Обучение модели {name} (поле 1)...")
                model = self.predictors1[name]
                if hasattr(model, 'set_lottery_params'):
                    model.set_lottery_params(
                        train_data.total_numbers,
                        train_data.pick_count,
                        train_data.total_numbers2 if train_data.is_double() else 0,
                        train_data.pick_count2 if train_data.is_double() else 0
                    )
                model.fit(blocks1_for_train)
                self.model_cache.save_model(self.profile_name, name, model, 1)
                trained_count += 1
                status = "✅" if getattr(model, 'is_trained', False) else "❌ (is_trained=False)"
                print(f"  [TRAIN] {name} обучена {status}")
            except Exception as e:
                print(f"[ERROR] Ошибка обучения {name} (поле 1): {e}")
                errors_count += 1

        # --- ОБУЧЕНИЕ ПОЛЯ 2 (если двойная лотерея) ---
        if train_data.is_double() and blocks2_for_train:
            # Используем те же имена моделей, что и для поля1 (без суффиксов)
            field2_names = [name for name in model_names if name in self.predictors2]
            if field2_names:
                print(f"[INFO] Будет обучено {len(field2_names)} моделей для поля 2")
            else:
                print("[WARN] Нет моделей для поля 2")

            for name in field2_names:
                if name not in self.predictors2:
                    print(f"[WARN] Модель {name} не найдена в predictors2")
                    continue

                if not force_retrain:
                    cached = self.model_cache.load_model(self.profile_name, name, 2)
                    if cached is not None:
                        print(f"[CACHE] Загружена модель {name} (поле 2)")
                        self.predictors2[name] = cached
                        loaded_count += 1
                        continue

                try:
                    print(f"[TRAIN] Обучение модели {name} (поле 2)...")
                    model = self.predictors2[name]
                    if hasattr(model, 'set_lottery_params'):
                        model.set_lottery_params(
                            train_data.total_numbers2,
                            train_data.pick_count2,
                            0, 0
                        )
                    model.fit(blocks2_for_train)
                    self.model_cache.save_model(self.profile_name, name, model, 2)
                    trained_count += 1
                    status = "✅" if getattr(model, 'is_trained', False) else "❌ (is_trained=False)"
                    print(f"  [TRAIN] {name} (поле 2) обучена {status}")
                except Exception as e:
                    print(f"[ERROR] Ошибка обучения {name} (поле 2): {e}")
                    errors_count += 1

        print("\n" + "=" * 50)
        print(f"[TRAIN] Итог: обучено {trained_count}, загружено из кэша {loaded_count}, ошибок {errors_count} из {total_tasks}")
        print("=" * 50)
        
    def _calculate_model_weights(self, blocks: List[List[int]], is_second_field: bool = False) -> Dict[str, float]:
        """
        Рассчитывает веса моделей на основе их точности на исторических данных.
        
        Args:
            blocks: Блоки для оценки
            is_second_field: True для второго поля, False для первого
        
        Returns:
            Словарь {имя_модели: вес}
        """
        import numpy as np

        if len(blocks) < 5:
            # Недостаточно данных для оценки — равные веса
            models = self.predictors2 if is_second_field else self.predictors1
            weights = {name: 1.0 for name in models.keys()
                       if name not in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble")}
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}

        # Разделяем данные для кросс-валидации
        split_idx = int(len(blocks) * 0.8)
        train_blocks = blocks[:split_idx]
        test_blocks = blocks[split_idx:]

        scores = {}
        models_to_evaluate = self.predictors2 if is_second_field else self.predictors1

        for name, model in models_to_evaluate.items():
            if name in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                continue

            try:
                # Обучаем на тренировочных данных
                model.fit(train_blocks)

                total_matches = 0
                valid_tests = 0

                for true_block in test_blocks:
                    try:
                        pred = model.predict_single()

                        if is_second_field:
                            # === ДЛЯ ВТОРОГО ПОЛЯ (1 число) ===
                            if isinstance(pred, tuple):
                                pred = pred[1] if len(pred) > 1 else pred[0]
                            if isinstance(pred, list):
                                pred = pred[0] if pred else None
                            elif isinstance(pred, (int, float)):
                                pred = int(pred)

                            if pred is None:
                                continue

                            true_num = true_block[0] if true_block else None
                            if true_num is not None and pred == true_num:
                                total_matches += 1
                            valid_tests += 1

                        else:
                            # === ДЛЯ ПЕРВОГО ПОЛЯ (несколько чисел) ===
                            if isinstance(pred, tuple):
                                pred = pred[0] if len(pred) > 0 else []
                            if not isinstance(pred, list) or len(pred) != self.data.pick_count:
                                continue

                            matches = len(set(pred) & set(true_block))
                            total_matches += matches
                            valid_tests += 1

                    except Exception:
                        continue

                if valid_tests > 0:
                    avg_matches = total_matches / valid_tests
                else:
                    avg_matches = 0.0

                # ===== ЗАЩИТА ОТ INF/NaN =====
                if not np.isfinite(avg_matches):
                    avg_matches = 0.0

                # Преобразуем в float и проверяем ещё раз
                avg_matches = float(avg_matches)
                if not np.isfinite(avg_matches):
                    avg_matches = 0.0

                # Минимальный вес 0.1, чтобы модели с нулём не выпадали
                scores[name] = max(avg_matches, 0.1)

            except Exception as e:
                print(f"  [WARN] Ошибка расчёта веса для {name}: {e}")
                scores[name] = 0.5  # Средний вес при ошибке

        # Удаляем возможные inf/NaN из scores
        for name in list(scores.keys()):
            if not np.isfinite(scores[name]):
                scores[name] = 0.1

        # Нормализуем веса
        total = sum(scores.values())
        if total > 0:
            weights = {k: v / total for k, v in scores.items()}
        else:
            # Fallback — равные веса
            weights = {k: 1.0 / len(scores) for k in scores}

        return weights

    def _create_ensembles(self, blocks1: List[List[int]], blocks2: List[List[int]] = None, force_retrain: bool = False, model_names: List[str] = None):
        """
        Создаёт ансамблевые модели для обоих полей и сохраняет их.
        # --- Локальные импорты ---
        try:
            from predictors.ensemble import EnsemblePredictor as SimpleEnsemble
        except ImportError:
            SimpleEnsemble = None
        try:
            from predictors.ensemble_voting import VotingEnsemblePredictor, WeightedEnsemblePredictor, StackingEnsemblePredictor
        except ImportError:
            VotingEnsemblePredictor = WeightedEnsemblePredictor = StackingEnsemblePredictor = None
        try:
            from predictors.blending import BlendingEnsemblePredictor
        except ImportError:
            BlendingEnsemblePredictor = None
        
        Args:
            blocks1: Блоки первого поля для обучения (от старых к новым)
            blocks2: Блоки второго поля для обучения (от старых к новым) или None
            force_retrain: Принудительно пересоздать ансамбли (игнорировать кэш)
        """
        print("\n[ENSEMBLE] Создание ансамблевых моделей...")
        trained_models = [name for name, p in self.predictors1.items() if p.is_trained]
        print(f"[DEBUG] Всего моделей в predictors1: {len(self.predictors1)}")
        print(f"[DEBUG] Обученных моделей: {len(trained_models)}")
        print(f"[DEBUG] Имена обученных: {trained_models[:10]}...")
        
        # --- Принудительное пересоздание: удаляем старые ансамбли из памяти и из кэша ---
        if force_retrain:
            for name in ["Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"]:
                if name in self.predictors1:
                    del self.predictors1[name]
                if name in self.predictors2:
                    del self.predictors2[name]
                if self.model_manager:
                    self.model_manager.delete_model(name, field=1)
                    if self.data.is_double():
                        self.model_manager.delete_model(name, field=2)
            print("  [INFO] Принудительное пересоздание ансамблей (кэш очищен)")
        
        # --- Проверка данных для первого поля ---
        base_models = {}
        for name, pred in self.predictors1.items():
            if name not in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                if pred.is_trained:
                    base_models[name] = pred
        
        if len(base_models) < 2:
            print(f"  [WARN] Недостаточно обученных моделей для ансамблей поле1: {len(base_models)}/2")
            return
        
        print(f"  [INFO] Найдено {len(base_models)} моделей для ансамблей поле1")
        
        # --- 1. Обычный ансамбль (Ensemble) для поля1 ---
        if "Ensemble" not in self.predictors1:
            # Пытаемся загрузить из кэша, если не force_retrain
            if not force_retrain and self.model_manager:
                loaded = self.model_manager.load_model("Ensemble", field=1)
                if loaded:
                    self.predictors1["Ensemble"] = loaded
                    print("  [LOAD] Ensemble (поле1) — загружен из кэша")
                else:
                    # Создаём новый
                    try:
                        from predictors.ensemble import EnsemblePredictor as SimpleEnsemble
                        
                        ensemble = SimpleEnsemble(base_models, self.data.pick_count, name="Ensemble")
                        ensemble.is_trained = True
                        
                        if self.data.is_double():
                            ensemble.is_double = True
                            ensemble.total_numbers2 = self.data.total_numbers2
                            ensemble.pick_count2 = self.data.pick_count2
                            if blocks2 and len(blocks2) > 0:
                                ensemble.blocks2 = blocks2
                        
                        self.predictors1["Ensemble"] = ensemble
                        if self.model_manager:
                            self.model_manager.save_model("Ensemble", ensemble, field=1)
                        print("  [OK] Ensemble (поле1) — создан и сохранён")
                    except Exception as e:
                        print(f"  [ERR] Ensemble (поле1): {e}")
            else:
                # Принудительно создаём заново
                try:
                    from predictors.ensemble import EnsemblePredictor as SimpleEnsemble
                    
                    ensemble = SimpleEnsemble(base_models, self.data.pick_count, name="Ensemble")
                    ensemble.is_trained = True
                    
                    if self.data.is_double():
                        ensemble.is_double = True
                        ensemble.total_numbers2 = self.data.total_numbers2
                        ensemble.pick_count2 = self.data.pick_count2
                        if blocks2 and len(blocks2) > 0:
                            ensemble.blocks2 = blocks2
                    
                    self.predictors1["Ensemble"] = ensemble
                    if self.model_manager:
                        self.model_manager.save_model("Ensemble", ensemble, field=1)
                    print("  [TRAIN] Ensemble (поле1) — создан и сохранён")
                except Exception as e:
                    print(f"  [ERR] Ensemble (поле1): {e}")
        
        # --- 2. Голосующие ансамбли для поля1 (если доступны) ---
        if HAS_VOTING:
            # Преобразуем в список для голосующих ансамблей
            base_models_list = list(base_models.values())
            
            if len(base_models_list) >= 2:
                
                # 2.1 Voting Ensemble (простое голосование)
                if "VotingEnsemble" not in self.predictors1:
                    if not force_retrain and self.model_manager:
                        loaded = self.model_manager.load_model("VotingEnsemble", field=1)
                        if loaded:
                            self.predictors1["VotingEnsemble"] = loaded
                            print("  [LOAD] VotingEnsemble (поле1) — загружен из кэша")
                        else:
                            try:
                                from predictors.ensemble_voting import VotingEnsemblePredictor
                                
                                voting = VotingEnsemblePredictor(
                                    total_numbers=self.data.total_numbers,
                                    pick_count=self.data.pick_count,
                                    predictors=base_models_list
                                )
                                
                                if self.data.is_double():
                                    voting.is_double = True
                                    voting.total_numbers2 = self.data.total_numbers2
                                    voting.pick_count2 = self.data.pick_count2
                                    if blocks2:
                                        voting.blocks2 = blocks2
                                        voting.predictors2 = list(self.predictors2.values())
                                
                                voting.fit(blocks1)
                                self.predictors1["VotingEnsemble"] = voting
                                if self.model_manager:
                                    self.model_manager.save_model("VotingEnsemble", voting, field=1)
                                print("  [OK] VotingEnsemble (поле1) — создан и сохранён")
                            except Exception as e:
                                print(f"  [ERR] VotingEnsemble (поле1): {e}")
                    else:
                        # Принудительно создаём заново
                        try:
                            from predictors.ensemble_voting import VotingEnsemblePredictor
                            
                            voting = VotingEnsemblePredictor(
                                total_numbers=self.data.total_numbers,
                                pick_count=self.data.pick_count,
                                predictors=base_models_list
                            )
                            
                            if self.data.is_double():
                                voting.is_double = True
                                voting.total_numbers2 = self.data.total_numbers2
                                voting.pick_count2 = self.data.pick_count2
                                if blocks2:
                                    voting.blocks2 = blocks2
                                    voting.predictors2 = list(self.predictors2.values())
                            
                            voting.fit(blocks1)
                            self.predictors1["VotingEnsemble"] = voting
                            if self.model_manager:
                                self.model_manager.save_model("VotingEnsemble", voting, field=1)
                            print("  [TRAIN] VotingEnsemble (поле1) — создан и сохранён")
                        except Exception as e:
                            print(f"  [ERR] VotingEnsemble (поле1): {e}")
                
                # 2.2 Weighted Ensemble (взвешенное голосование)
                if "WeightedEnsemble" not in self.predictors1:
                    if not force_retrain and self.model_manager:
                        loaded = self.model_manager.load_model("WeightedEnsemble", field=1)
                        if loaded:
                            self.predictors1["WeightedEnsemble"] = loaded
                            print("  [LOAD] WeightedEnsemble (поле1) — загружен из кэша")
                        else:
                            try:
                                from predictors.ensemble_voting import WeightedEnsemblePredictor
                                from core.parameter_manager import ParameterManager
                                pm = ParameterManager(self.profile_path)
                                performance_based = pm.get_param("WeightedEnsemble", "performance_based", True)
                                min_weight = pm.get_param("WeightedEnsemble", "min_weight", 0.01)
                                use_ranking = pm.get_param("WeightedEnsemble", "use_ranking", False)
                                base_models_config = pm.get_param("WeightedEnsemble", "base_models", [])

                                # Формируем список базовых моделей в соответствии с конфигурацией
                                if base_models_config == "top10":
                                    weights = self._calculate_model_weights(blocks1)
                                    # сортируем по убыванию веса
                                    top_names = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
                                    base_models_list = [self.predictors1[name] for name, _ in top_names if name in self.predictors1]
                                    print(f"  [INFO] WeightedEnsemble: выбрано топ-10 моделей (из {len(weights)})")
                                elif isinstance(base_models_config, list) and base_models_config:
                                    # Явный список имён
                                    base_models_list = []
                                    for name in base_models_config:
                                        if name in self.predictors1:
                                            base_models_list.append(self.predictors1[name])
                                        else:
                                            print(f"  [WARN] Модель {name} не найдена, пропускаем")
                                    print(f"  [INFO] WeightedEnsemble: выбрано {len(base_models_list)} моделей из списка")
                                else:
                                    # По умолчанию – все не-ансамбли
                                    base_models_list = [m for name, m in self.predictors1.items()
                                                        if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                    print(f"  [INFO] WeightedEnsemble: выбраны все не-ансамблевые модели ({len(base_models_list)})")

                                if not base_models_list:
                                    print("  [WARN] WeightedEnsemble: нет базовых моделей, пропускаем")
                                else:
                                    weights = self._calculate_model_weights(blocks1)
                                    weighted = WeightedEnsemblePredictor(
                                        total_numbers=self.data.total_numbers,
                                        pick_count=self.data.pick_count,
                                        predictors=base_models_list,
                                        weights=weights,
                                        name="WeightedEnsemble",
                                        performance_based=performance_based,
                                        min_weight=min_weight,
                                        use_ranking=use_ranking
                                    )
                                    if self.data.is_double():
                                        weighted.is_double = True
                                        weighted.total_numbers2 = self.data.total_numbers2
                                        weighted.pick_count2 = self.data.pick_count2
                                        if blocks2:
                                            weighted.blocks2 = blocks2
                                            # Для второго поля тоже нужно выбрать модели
                                            base_models_config2 = pm.get_param("WeightedEnsemble", "base_models", [])
                                            if base_models_config2 == "top10":
                                                weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                                top_names2 = sorted(weights2.items(), key=lambda x: x[1], reverse=True)[:10]
                                                base_models2 = [self.predictors2[name] for name, _ in top_names2 if name in self.predictors2]
                                            elif isinstance(base_models_config2, list) and base_models_config2:
                                                base_models2 = [self.predictors2[name] for name in base_models_config2 if name in self.predictors2]
                                            else:
                                                base_models2 = [m for name, m in self.predictors2.items()
                                                                if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                            weighted.predictors2 = base_models2
                                            weighted.weights2 = weights  # можно пересчитать отдельно
                                    weighted.fit(blocks1)
                                    self.predictors1["WeightedEnsemble"] = weighted
                                    if self.model_manager:
                                        self.model_manager.save_model("WeightedEnsemble", weighted, field=1)
                                    print("  [OK] WeightedEnsemble (поле1) — создан и сохранён")
                            except Exception as e:
                                print(f"  [ERR] WeightedEnsemble (поле1): {e}")
                    else:
                        try:
                            from predictors.ensemble_voting import WeightedEnsemblePredictor
                            from core.parameter_manager import ParameterManager
                            pm = ParameterManager(self.profile_path)
                            performance_based = pm.get_param("WeightedEnsemble", "performance_based", True)
                            min_weight = pm.get_param("WeightedEnsemble", "min_weight", 0.01)
                            use_ranking = pm.get_param("WeightedEnsemble", "use_ranking", False)
                            base_models_config = pm.get_param("WeightedEnsemble", "base_models", [])

                            # Формируем список базовых моделей в соответствии с конфигурацией
                            if base_models_config == "top10":
                                weights = self._calculate_model_weights(blocks1)
                                top_names = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
                                base_models_list = [self.predictors1[name] for name, _ in top_names if name in self.predictors1]
                                print(f"  [INFO] WeightedEnsemble: выбрано топ-10 моделей (из {len(weights)})")
                            elif isinstance(base_models_config, list) and base_models_config:
                                base_models_list = [self.predictors1[name] for name in base_models_config if name in self.predictors1]
                                print(f"  [INFO] WeightedEnsemble: выбрано {len(base_models_list)} моделей из списка")
                            else:
                                base_models_list = [m for name, m in self.predictors1.items()
                                                    if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                print(f"  [INFO] WeightedEnsemble: выбраны все не-ансамблевые модели ({len(base_models_list)})")

                            if not base_models_list:
                                print("  [WARN] WeightedEnsemble: нет базовых моделей, пропускаем")
                            else:
                                weights = self._calculate_model_weights(blocks1)
                                weighted = WeightedEnsemblePredictor(
                                    total_numbers=self.data.total_numbers,
                                    pick_count=self.data.pick_count,
                                    predictors=base_models_list,
                                    weights=weights,
                                    name="WeightedEnsemble",
                                    performance_based=performance_based,
                                    min_weight=min_weight,
                                    use_ranking=use_ranking
                                )
                                if self.data.is_double():
                                    weighted.is_double = True
                                    weighted.total_numbers2 = self.data.total_numbers2
                                    weighted.pick_count2 = self.data.pick_count2
                                    if blocks2:
                                        weighted.blocks2 = blocks2
                                        base_models_config2 = pm.get_param("WeightedEnsemble", "base_models", [])
                                        if base_models_config2 == "top10":
                                            weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                            top_names2 = sorted(weights2.items(), key=lambda x: x[1], reverse=True)[:10]
                                            base_models2 = [self.predictors2[name] for name, _ in top_names2 if name in self.predictors2]
                                        elif isinstance(base_models_config2, list) and base_models_config2:
                                            base_models2 = [self.predictors2[name] for name in base_models_config2 if name in self.predictors2]
                                        else:
                                            base_models2 = [m for name, m in self.predictors2.items()
                                                            if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                        weighted.predictors2 = base_models2
                                        weighted.weights2 = weights
                                weighted.fit(blocks1)
                                self.predictors1["WeightedEnsemble"] = weighted
                                if self.model_manager:
                                    self.model_manager.save_model("WeightedEnsemble", weighted, field=1)
                                print("  [TRAIN] WeightedEnsemble (поле1) — создан и сохранён")
                        except Exception as e:
                            print(f"  [ERR] WeightedEnsemble (поле1): {e}")
                
                # 2.3 Stacking Ensemble (стэкинг)
                if "StackingEnsemble" not in self.predictors1:
                    if not force_retrain and self.model_manager:
                        loaded = self.model_manager.load_model("StackingEnsemble", field=1)
                        if loaded:
                            self.predictors1["StackingEnsemble"] = loaded
                            print("  [LOAD] StackingEnsemble (поле1) — загружен из кэша")
                        else:
                            try:
                                from predictors.ensemble_voting import StackingEnsemblePredictor
                                
                                top_models = base_models_list[:5]
                                
                                stacking = StackingEnsemblePredictor(
                                    total_numbers=self.data.total_numbers,
                                    pick_count=self.data.pick_count,
                                    base_predictors=top_models
                                )
                                
                                if self.data.is_double():
                                    stacking.is_double = True
                                    stacking.total_numbers2 = self.data.total_numbers2
                                    stacking.pick_count2 = self.data.pick_count2
                                    if blocks2:
                                        stacking.blocks2 = blocks2
                                        stacking.base_predictors2 = list(self.predictors2.values())[:5]
                                
                                stacking.fit(blocks1)
                                self.predictors1["StackingEnsemble"] = stacking
                                if self.model_manager:
                                    self.model_manager.save_model("StackingEnsemble", stacking, field=1)
                                print("  [OK] StackingEnsemble (поле1) — создан и сохранён")
                            except Exception as e:
                                print(f"  [ERR] StackingEnsemble (поле1): {e}")
                    else:
                        try:
                            from predictors.ensemble_voting import StackingEnsemblePredictor
                            
                            top_models = base_models_list[:5]
                            
                            stacking = StackingEnsemblePredictor(
                                total_numbers=self.data.total_numbers,
                                pick_count=self.data.pick_count,
                                base_predictors=top_models
                            )
                            
                            if self.data.is_double():
                                stacking.is_double = True
                                stacking.total_numbers2 = self.data.total_numbers2
                                stacking.pick_count2 = self.data.pick_count2
                                if blocks2:
                                    stacking.blocks2 = blocks2
                                    stacking.base_predictors2 = list(self.predictors2.values())[:5]
                            
                            stacking.fit(blocks1)
                            self.predictors1["StackingEnsemble"] = stacking
                            if self.model_manager:
                                self.model_manager.save_model("StackingEnsemble", stacking, field=1)
                            print("  [TRAIN] StackingEnsemble (поле1) — создан и сохранён")
                        except Exception as e:
                            print(f"  [ERR] StackingEnsemble (поле1): {e}")
        
        # --- АНСАМБЛИ ДЛЯ ВТОРОГО ПОЛЯ (если двойная лотерея) ---
        if blocks2 is not None and self.data.is_double() and len(blocks2) > 0:
            print("\n  [FIELD2] Создание ансамблей для второго поля...")
            
            # Собираем обученные модели для второго поля
            base_models2 = {}
            for name, pred in self.predictors2.items():
                if name not in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                    if pred.is_trained:
                        base_models2[name] = pred
            
            if len(base_models2) < 2:
                print(f"  [WARN] Недостаточно моделей для ансамблей поле2: {len(base_models2)}/2")
            else:
                print(f"  [INFO] Найдено {len(base_models2)} моделей для ансамблей поле2")
                
                # 1. Обычный ансамбль для поля2
                if "Ensemble" not in self.predictors2:
                    if not force_retrain and self.model_manager:
                        loaded = self.model_manager.load_model("Ensemble", field=2)
                        if loaded:
                            self.predictors2["Ensemble"] = loaded
                            print("  [LOAD] Ensemble (поле2) — загружен из кэша")
                        else:
                            try:
                                from predictors.ensemble import EnsemblePredictor as SimpleEnsemble
                                
                                ensemble2 = SimpleEnsemble(base_models2, self.data.pick_count2, name="Ensemble")
                                ensemble2.is_trained = True
                                ensemble2.is_double = True
                                self.predictors2["Ensemble"] = ensemble2
                                if self.model_manager:
                                    self.model_manager.save_model("Ensemble", ensemble2, field=2)
                                print("  [OK] Ensemble (поле2) — создан и сохранён")
                            except Exception as e:
                                print(f"  [ERR] Ensemble (поле2): {e}")
                    else:
                        try:
                            from predictors.ensemble import EnsemblePredictor as SimpleEnsemble
                            
                            ensemble2 = SimpleEnsemble(base_models2, self.data.pick_count2, name="Ensemble")
                            ensemble2.is_trained = True
                            ensemble2.is_double = True
                            self.predictors2["Ensemble"] = ensemble2
                            if self.model_manager:
                                self.model_manager.save_model("Ensemble", ensemble2, field=2)
                            print("  [TRAIN] Ensemble (поле2) — создан и сохранён")
                        except Exception as e:
                            print(f"  [ERR] Ensemble (поле2): {e}")
                
                # 2. Голосующие ансамбли для поля2
                if HAS_VOTING:
                    base_models2_list = list(base_models2.values())
                    
                    if len(base_models2_list) >= 2:
                        
                        # Voting Ensemble для поля2
                        if "VotingEnsemble" not in self.predictors2:
                            if not force_retrain and self.model_manager:
                                loaded = self.model_manager.load_model("VotingEnsemble", field=2)
                                if loaded:
                                    self.predictors2["VotingEnsemble"] = loaded
                                    print("  [LOAD] VotingEnsemble (поле2) — загружен из кэша")
                                else:
                                    try:
                                        from predictors.ensemble_voting import VotingEnsemblePredictor
                                        
                                        voting2 = VotingEnsemblePredictor(
                                            total_numbers=self.data.total_numbers2,
                                            pick_count=self.data.pick_count2,
                                            predictors=base_models2_list
                                        )
                                        voting2.is_double = True
                                        voting2.total_numbers2 = self.data.total_numbers2
                                        voting2.pick_count2 = self.data.pick_count2
                                        voting2.fit(blocks2)
                                        self.predictors2["VotingEnsemble"] = voting2
                                        if self.model_manager:
                                            self.model_manager.save_model("VotingEnsemble", voting2, field=2)
                                        print("  [OK] VotingEnsemble (поле2) — создан и сохранён")
                                    except Exception as e:
                                        print(f"  [ERR] VotingEnsemble (поле2): {e}")
                            else:
                                try:
                                    from predictors.ensemble_voting import VotingEnsemblePredictor
                                    
                                    voting2 = VotingEnsemblePredictor(
                                        total_numbers=self.data.total_numbers2,
                                        pick_count=self.data.pick_count2,
                                        predictors=base_models2_list
                                    )
                                    voting2.is_double = True
                                    voting2.total_numbers2 = self.data.total_numbers2
                                    voting2.pick_count2 = self.data.pick_count2
                                    voting2.fit(blocks2)
                                    self.predictors2["VotingEnsemble"] = voting2
                                    if self.model_manager:
                                        self.model_manager.save_model("VotingEnsemble", voting2, field=2)
                                    print("  [TRAIN] VotingEnsemble (поле2) — создан и сохранён")
                                except Exception as e:
                                    print(f"  [ERR] VotingEnsemble (поле2): {e}")
                        
                        # Weighted Ensemble для поля2
                        if "WeightedEnsemble" not in self.predictors2:
                            if not force_retrain and self.model_manager:
                                loaded = self.model_manager.load_model("WeightedEnsemble", field=2)
                                if loaded:
                                    self.predictors2["WeightedEnsemble"] = loaded
                                    print("  [LOAD] WeightedEnsemble (поле2) — загружен из кэша")
                                else:
                                    try:
                                        from predictors.ensemble_voting import WeightedEnsemblePredictor
                                        from core.parameter_manager import ParameterManager
                                        pm = ParameterManager(self.profile_path)
                                        performance_based = pm.get_param("WeightedEnsemble", "performance_based", True)
                                        min_weight = pm.get_param("WeightedEnsemble", "min_weight", 0.01)
                                        use_ranking = pm.get_param("WeightedEnsemble", "use_ranking", False)
                                        base_models_config = pm.get_param("WeightedEnsemble", "base_models", [])

                                        # Формируем список базовых моделей для поля2
                                        if base_models_config == "top10":
                                            weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                            top_names = sorted(weights2.items(), key=lambda x: x[1], reverse=True)[:10]
                                            base_models2_list = [self.predictors2[name] for name, _ in top_names if name in self.predictors2]
                                            print(f"  [INFO] WeightedEnsemble (поле2): выбрано топ-10 моделей (из {len(weights2)})")
                                        elif isinstance(base_models_config, list) and base_models_config:
                                            base_models2_list = [self.predictors2[name] for name in base_models_config if name in self.predictors2]
                                            print(f"  [INFO] WeightedEnsemble (поле2): выбрано {len(base_models2_list)} моделей из списка")
                                        else:
                                            # По умолчанию – все не-ансамбли
                                            base_models2_list = [m for name, m in self.predictors2.items() 
                                                                 if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                            print(f"  [INFO] WeightedEnsemble (поле2): выбраны все не-ансамблевые модели ({len(base_models2_list)})")

                                        if not base_models2_list:
                                            print("  [WARN] WeightedEnsemble (поле2): нет базовых моделей, пропускаем")
                                        else:
                                            weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                            weighted2 = WeightedEnsemblePredictor(
                                                total_numbers=self.data.total_numbers2,
                                                pick_count=self.data.pick_count2,
                                                predictors=base_models2_list,
                                                weights=weights2,
                                                name="WeightedEnsemble",
                                                performance_based=performance_based,
                                                min_weight=min_weight,
                                                use_ranking=use_ranking
                                            )
                                            weighted2.is_double = True
                                            weighted2.total_numbers2 = self.data.total_numbers2
                                            weighted2.pick_count2 = self.data.pick_count2
                                            weighted2.fit(blocks2)
                                            self.predictors2["WeightedEnsemble"] = weighted2
                                            if self.model_manager:
                                                self.model_manager.save_model("WeightedEnsemble", weighted2, field=2)
                                            print("  [OK] WeightedEnsemble (поле2) — создан и сохранён")
                                    except Exception as e:
                                        print(f"  [ERR] WeightedEnsemble (поле2): {e}")
                            else:
                                try:
                                    from predictors.ensemble_voting import WeightedEnsemblePredictor
                                    from core.parameter_manager import ParameterManager
                                    pm = ParameterManager(self.profile_path)
                                    performance_based = pm.get_param("WeightedEnsemble", "performance_based", True)
                                    min_weight = pm.get_param("WeightedEnsemble", "min_weight", 0.01)
                                    use_ranking = pm.get_param("WeightedEnsemble", "use_ranking", False)
                                    base_models_config = pm.get_param("WeightedEnsemble", "base_models", [])

                                    # Формируем список базовых моделей для поля2
                                    if base_models_config == "top10":
                                        weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                        top_names = sorted(weights2.items(), key=lambda x: x[1], reverse=True)[:10]
                                        base_models2_list = [self.predictors2[name] for name, _ in top_names if name in self.predictors2]
                                        print(f"  [INFO] WeightedEnsemble (поле2): выбрано топ-10 моделей (из {len(weights2)})")
                                    elif isinstance(base_models_config, list) and base_models_config:
                                        base_models2_list = [self.predictors2[name] for name in base_models_config if name in self.predictors2]
                                        print(f"  [INFO] WeightedEnsemble (поле2): выбрано {len(base_models2_list)} моделей из списка")
                                    else:
                                        base_models2_list = [m for name, m in self.predictors2.items() 
                                                             if not name.endswith("Ensemble") and name != "WeightedEnsemble"]
                                        print(f"  [INFO] WeightedEnsemble (поле2): выбраны все не-ансамблевые модели ({len(base_models2_list)})")

                                    if not base_models2_list:
                                        print("  [WARN] WeightedEnsemble (поле2): нет базовых моделей, пропускаем")
                                    else:
                                        weights2 = self._calculate_model_weights(blocks2, is_second_field=True)
                                        weighted2 = WeightedEnsemblePredictor(
                                            total_numbers=self.data.total_numbers2,
                                            pick_count=self.data.pick_count2,
                                            predictors=base_models2_list,
                                            weights=weights2,
                                            name="WeightedEnsemble",
                                            performance_based=performance_based,
                                            min_weight=min_weight,
                                            use_ranking=use_ranking
                                        )
                                        weighted2.is_double = True
                                        weighted2.total_numbers2 = self.data.total_numbers2
                                        weighted2.pick_count2 = self.data.pick_count2
                                        weighted2.fit(blocks2)
                                        self.predictors2["WeightedEnsemble"] = weighted2
                                        if self.model_manager:
                                            self.model_manager.save_model("WeightedEnsemble", weighted2, field=2)
                                        print("  [TRAIN] WeightedEnsemble (поле2) — создан и сохранён")
                                except Exception as e:
                                    print(f"  [ERR] WeightedEnsemble (поле2): {e}")
                        
                        # Stacking Ensemble для поля2
                        if "StackingEnsemble" not in self.predictors2:
                            if not force_retrain and self.model_manager:
                                loaded = self.model_manager.load_model("StackingEnsemble", field=2)
                                if loaded:
                                    self.predictors2["StackingEnsemble"] = loaded
                                    print("  [LOAD] StackingEnsemble (поле2) — загружен из кэша")
                                else:
                                    try:
                                        from predictors.ensemble_voting import StackingEnsemblePredictor
                                        
                                        top_models2 = base_models2_list[:3]
                                        
                                        stacking2 = StackingEnsemblePredictor(
                                            total_numbers=self.data.total_numbers2,
                                            pick_count=self.data.pick_count2,
                                            base_predictors=top_models2
                                        )
                                        stacking2.is_double = True
                                        stacking2.total_numbers2 = self.data.total_numbers2
                                        stacking2.pick_count2 = self.data.pick_count2
                                        stacking2.fit(blocks2)
                                        self.predictors2["StackingEnsemble"] = stacking2
                                        if self.model_manager:
                                            self.model_manager.save_model("StackingEnsemble", stacking2, field=2)
                                        print("  [OK] StackingEnsemble (поле2) — создан и сохранён")
                                    except Exception as e:
                                        print(f"  [ERR] StackingEnsemble (поле2): {e}")
                            else:
                                try:
                                    from predictors.ensemble_voting import StackingEnsemblePredictor
                                    
                                    top_models2 = base_models2_list[:3]
                                    
                                    stacking2 = StackingEnsemblePredictor(
                                        total_numbers=self.data.total_numbers2,
                                        pick_count=self.data.pick_count2,
                                        base_predictors=top_models2
                                    )
                                    stacking2.is_double = True
                                    stacking2.total_numbers2 = self.data.total_numbers2
                                    stacking2.pick_count2 = self.data.pick_count2
                                    stacking2.fit(blocks2)
                                    self.predictors2["StackingEnsemble"] = stacking2
                                    if self.model_manager:
                                        self.model_manager.save_model("StackingEnsemble", stacking2, field=2)
                                    print("  [TRAIN] StackingEnsemble (поле2) — создан и сохранён")
                                except Exception as e:
                                    print(f"  [ERR] StackingEnsemble (поле2): {e}")
                        # В конце метода _create_ensembles, после создания StackingEnsemble, добавьте:

        # =====================================================
        # 3.3 Blending Ensemble (блендинг) — для поля1
        # =====================================================
        if "BlendingEnsemble" not in self.predictors1:
            if not force_retrain and self.model_manager:
                loaded = self.model_manager.load_model("BlendingEnsemble", field=1)
                if loaded:
                    self.predictors1["BlendingEnsemble"] = loaded
                    print("  [LOAD] BlendingEnsemble (поле1) — загружен из кэша")
                else:
                    try:
                        from core.parameter_manager import ParameterManager
                        pm = ParameterManager(self.profile_path)
                        meta_model = pm.get_param("BlendingEnsemble", "meta_model", "ridge")
                        n_folds = pm.get_param("BlendingEnsemble", "n_folds", 5)
                        use_original_features = pm.get_param("BlendingEnsemble", "use_original_features", False)

                        # Базовые модели (исключаем ансамбли и себя)
                        base_models_list = [m for name, m in self.predictors1.items()
                                            if not name.endswith("Ensemble") and name != "BlendingEnsemble"]
                        base_models_list = [m for m in base_models_list if m.name != "BlendingEnsemble"]

                        if len(base_models_list) >= 3:
                            blending = BlendingEnsemblePredictor(
                                total_numbers=self.data.total_numbers,
                                pick_count=self.data.pick_count,
                                base_predictors=base_models_list,
                                meta_model_type=meta_model,
                                n_folds=n_folds,
                                use_original_features=use_original_features,
                                name="BlendingEnsemble"
                            )
                            blending.fit(blocks1)
                            self.predictors1["BlendingEnsemble"] = blending
                            if self.model_manager:
                                self.model_manager.save_model("BlendingEnsemble", blending, field=1)
                            print("  [OK] BlendingEnsemble (поле1) — создан и сохранён")
                        else:
                            print("  [WARN] BlendingEnsemble (поле1): недостаточно базовых моделей (минимум 3)")
                    except Exception as e:
                        print(f"  [ERR] BlendingEnsemble (поле1): {e}")
            else:
                try:
                    from core.parameter_manager import ParameterManager
                    pm = ParameterManager(self.profile_path)
                    meta_model = pm.get_param("BlendingEnsemble", "meta_model", "ridge")
                    n_folds = pm.get_param("BlendingEnsemble", "n_folds", 5)
                    use_original_features = pm.get_param("BlendingEnsemble", "use_original_features", False)

                    base_models_list = [m for name, m in self.predictors1.items()
                                        if not name.endswith("Ensemble") and name != "BlendingEnsemble"]
                    base_models_list = [m for m in base_models_list if m.name != "BlendingEnsemble"]

                    if len(base_models_list) >= 3:
                        blending = BlendingEnsemblePredictor(
                            total_numbers=self.data.total_numbers,
                            pick_count=self.data.pick_count,
                            base_predictors=base_models_list,
                            meta_model_type=meta_model,
                            n_folds=n_folds,
                            use_original_features=use_original_features,
                            name="BlendingEnsemble"
                        )
                        blending.fit(blocks1)
                        self.predictors1["BlendingEnsemble"] = blending
                        if self.model_manager:
                            self.model_manager.save_model("BlendingEnsemble", blending, field=1)
                        print("  [TRAIN] BlendingEnsemble (поле1) — создан и сохранён")
                    else:
                        print("  [WARN] BlendingEnsemble (поле1): недостаточно базовых моделей (минимум 3)")
                except Exception as e:
                    print(f"  [ERR] BlendingEnsemble (поле1): {e}")

        # =====================================================
        # Blending Ensemble для поля2 (если двойная лотерея)
        # =====================================================
        if self.data.is_double() and blocks2 is not None and len(blocks2) > 0:
            if "BlendingEnsemble" not in self.predictors2:
                if not force_retrain and self.model_manager:
                    loaded = self.model_manager.load_model("BlendingEnsemble", field=2)
                    if loaded:
                        self.predictors2["BlendingEnsemble"] = loaded
                        print("  [LOAD] BlendingEnsemble (поле2) — загружен из кэша")
                    else:
                        try:
                            from core.parameter_manager import ParameterManager
                            pm = ParameterManager(self.profile_path)
                            meta_model = pm.get_param("BlendingEnsemble", "meta_model", "ridge")
                            n_folds = pm.get_param("BlendingEnsemble", "n_folds", 5)
                            use_original_features = pm.get_param("BlendingEnsemble", "use_original_features", False)

                            base_models2_list = [m for name, m in self.predictors2.items()
                                                 if not name.endswith("Ensemble") and name != "BlendingEnsemble"]
                            base_models2_list = [m for m in base_models2_list if m.name != "BlendingEnsemble"]

                            if len(base_models2_list) >= 3:
                                blending2 = BlendingEnsemblePredictor(
                                    total_numbers=self.data.total_numbers2,
                                    pick_count=self.data.pick_count2,
                                    base_predictors=base_models2_list,
                                    meta_model_type=meta_model,
                                    n_folds=n_folds,
                                    use_original_features=use_original_features,
                                    name="BlendingEnsemble"
                                )
                                blending2.fit(blocks2)
                                self.predictors2["BlendingEnsemble"] = blending2
                                if self.model_manager:
                                    self.model_manager.save_model("BlendingEnsemble", blending2, field=2)
                                print("  [OK] BlendingEnsemble (поле2) — создан и сохранён")
                            else:
                                print("  [WARN] BlendingEnsemble (поле2): недостаточно базовых моделей (минимум 3)")
                        except Exception as e:
                            print(f"  [ERR] BlendingEnsemble (поле2): {e}")
                else:
                    try:
                        from core.parameter_manager import ParameterManager
                        pm = ParameterManager(self.profile_path)
                        meta_model = pm.get_param("BlendingEnsemble", "meta_model", "ridge")
                        n_folds = pm.get_param("BlendingEnsemble", "n_folds", 5)
                        use_original_features = pm.get_param("BlendingEnsemble", "use_original_features", False)

                        base_models2_list = [m for name, m in self.predictors2.items()
                                             if not name.endswith("Ensemble") and name != "BlendingEnsemble"]
                        base_models2_list = [m for m in base_models2_list if m.name != "BlendingEnsemble"]

                        if len(base_models2_list) >= 3:
                            blending2 = BlendingEnsemblePredictor(
                                total_numbers=self.data.total_numbers2,
                                pick_count=self.data.pick_count2,
                                base_predictors=base_models2_list,
                                meta_model_type=meta_model,
                                n_folds=n_folds,
                                use_original_features=use_original_features,
                                name="BlendingEnsemble"
                            )
                            blending2.fit(blocks2)
                            self.predictors2["BlendingEnsemble"] = blending2
                            if self.model_manager:
                                self.model_manager.save_model("BlendingEnsemble", blending2, field=2)
                            print("  [TRAIN] BlendingEnsemble (поле2) — создан и сохранён")
                        else:
                            print("  [WARN] BlendingEnsemble (поле2): недостаточно базовых моделей (минимум 3)")
                    except Exception as e:
                        print(f"  [ERR] BlendingEnsemble (поле2): {e}")
        
        print("  [DONE] Создание ансамблей завершено")

    def predict_all_models(self) -> Dict[str, Tuple[List[int], List[int]]]:
        """Возвращает для каждой модели пару (прогноз1, прогноз2)."""
        result = {}
        
        if not self.data.is_double():
            # Обычная лотерея
            for name, pred in self.predictors1.items():
                try:
                    p = pred.predict_single()
                    if p and isinstance(p, list):
                        result[name] = (p, [])
                    else:
                        result[name] = ([], [])
                except Exception:
                    result[name] = ([], [])
            return result
        
        # Двойная лотерея
        all_names = set(self.predictors1.keys()) | set(self.predictors2.keys())
        
        # Список моделей, которые нужно обработать отдельно
        problematic_models = ["Bayesian", "Bootstrap"]
        
        for name in all_names:
            p1 = []
            p2 = []
            
            # --- Явная обработка для проблемных моделей ---
            if name in problematic_models:
                # Для первого поля — частотный анализ
                try:
                    from collections import Counter
                    # Первое поле
                    counter1 = Counter()
                    for block in self.data.blocks:
                        counter1.update(block)
                    top1 = [num for num, _ in counter1.most_common(self.data.pick_count)]
                    top1.sort()
                    p1 = top1
                    
                    # Второе поле
                    if self.data.blocks2:
                        counter2 = Counter()
                        for block in self.data.blocks2:
                            if block:
                                counter2.update(block)
                        if counter2:
                            p2 = [counter2.most_common(1)[0][0]]
                        else:
                            p2 = [self.data.total_numbers2 // 2]
                    else:
                        p2 = [self.data.total_numbers2 // 2]
                        
                    result[name] = (p1, p2)
                    continue  # Переходим к следующей модели
                except Exception as e:
                    print(f"[WARN] Ошибка при обработке {name}: {e}")
                    # fallback
                    p1 = list(range(1, min(self.data.pick_count + 1, self.data.total_numbers + 1)))
                    p2 = [self.data.total_numbers2 // 2]
                    result[name] = (p1, p2)
                    continue
            
            # --- Обычная обработка для остальных моделей ---
            
            # Получаем прогноз первого поля
            if name in self.predictors1:
                try:
                    pred = self.predictors1[name].predict_single()
                    if isinstance(pred, tuple) and len(pred) == 2:
                        p1 = pred[0] if pred[0] else []
                        if not p2 and pred[1]:
                            p2 = pred[1]
                    elif isinstance(pred, list):
                        p1 = pred
                    elif isinstance(pred, (int, float)):
                        p1 = [int(pred)]
                except Exception:
                    p1 = []
            
            # Получаем прогноз второго поля
            if name in self.predictors2:
                try:
                    pred = self.predictors2[name].predict_single()
                    if isinstance(pred, list):
                        p2 = pred
                    elif isinstance(pred, (int, float)):
                        p2 = [int(pred)]
                    elif isinstance(pred, tuple) and len(pred) == 2 and pred[1]:
                        p2 = pred[1]
                except Exception:
                    p2 = []
            
            # Если второе поле не получено, используем частотный анализ
            if not p2 and name in self.predictors2:
                try:
                    if hasattr(self.predictors2[name], 'blocks') and self.predictors2[name].blocks:
                        from collections import Counter
                        counter = Counter()
                        for block in self.predictors2[name].blocks:
                            if block:
                                counter.update(block)
                        if counter:
                            p2 = [counter.most_common(1)[0][0]]
                except Exception:
                    pass
            
            # Если всё ещё нет второго поля — заглушка
            if not p2:
                p2 = [self.data.total_numbers2 // 2]
            
            # Если нет первого поля — заглушка
            if not p1:
                p1 = list(range(1, min(self.data.pick_count + 1, self.data.total_numbers + 1)))
            
            # Преобразуем в списки если нужно
            if not isinstance(p1, list):
                p1 = [p1] if p1 else []
            if not isinstance(p2, list):
                p2 = [p2] if p2 else []
            
            result[name] = (p1, p2)
        
        return result
        
        

    def predict_top_n(self, n: int = 3) -> List[Tuple[List[int], List[int], float]]:
        if self.data.blocks_count() < 10:
            print("[WARN] Недостаточно данных для расчёта уверенности (<10 тиражей)")
            weights = {name: 1.0 for name in self.predictors1.keys() 
                       if name not in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble")}
            print("[INFO] Используются равные веса для всех моделей")
        else:
            weights = self.evaluate_models(test_ratio=0.2)
            
        if not weights:
            print("[WARN] Не удалось рассчитать веса моделей, используем равные")
            weights = {name: 1.0 for name in self.predictors1.keys() 
                       if name not in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble")}
        
        # ==================================================================
        # ОДИНАРНАЯ ЛОТЕРЕЯ (одно поле)
        # ==================================================================
        if not self.data.is_double():
            all_predictions = {}
            for name, model in self.predictors1.items():
                if name in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                    continue
                confidence = weights.get(name, 0.0)
                if confidence >= 0:
                    try:
                        pred = model.predict_single()
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        if pred and len(pred) == self.data.pick_count:
                            all_predictions[tuple(pred)] = all_predictions.get(tuple(pred), 0) + confidence
                    except Exception:
                        pass
            
            # Fallback, если нет предсказаний
            if not all_predictions:
                print("[WARN] Нет предсказаний, используем Frequency")
                freq_class = MODEL_REGISTRY.get("Frequency")
                if freq_class is None:
                    from predictors.frequency import FrequencyPredictor
                    freq_class = FrequencyPredictor
                freq_model = freq_class(...)
                blocks_for_train = list(reversed(self.data.blocks))
                freq_model.fit(blocks_for_train)
                pred = freq_model.predict_single()
                if pred:
                    all_predictions = {tuple(pred): 1.0}
                else:
                    all_predictions = {tuple(range(1, self.data.pick_count + 1)): 1.0}
            
            max_weight = max(all_predictions.values()) if all_predictions else 1
            top_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:n]
            
            results = []
            for i in range(len(top_preds)):
                p = list(top_preds[i][0])
                p.sort()
                conf = (top_preds[i][1] / max_weight) * 100
                if conf > 80:
                    conf = 80.0
                # Для одинарной лотереи поле2 – пустой список
                results.append((p, [], round(conf, 1)))
            
            return results
        
        # ==================================================================
        # ДВОЙНАЯ ЛОТЕРЕЯ (два поля)
        # ==================================================================
        # === ПОЛЕ 1 ===
        all_predictions1 = {}
        for name, model in self.predictors1.items():
            if name in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                continue
            confidence = weights.get(name, 0.0)
            if confidence >= 0:
                try:
                    pred = model.predict_single()
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if pred and len(pred) == self.data.pick_count:
                        all_predictions1[tuple(pred)] = all_predictions1.get(tuple(pred), 0) + confidence
                except Exception:
                    pass
        
        # Fallback для поля1
        if not all_predictions1:
            print("[WARN] Нет предсказаний для поля1, используем Frequency")
            freq_class = MODEL_REGISTRY.get("Frequency")
            if freq_class is None:
                from predictors.frequency import FrequencyPredictor
                freq_class = FrequencyPredictor
            freq_model = freq_class(...)
            blocks_for_train = list(reversed(self.data.blocks))
            freq_model.fit(blocks_for_train)
            pred = freq_model.predict_single()
            if pred:
                all_predictions1 = {tuple(pred): 1.0}
            else:
                all_predictions1 = {tuple(range(1, self.data.pick_count + 1)): 1.0}
        
        # === ПОЛЕ 2 ===
        all_predictions2 = {}
        for name, model in self.predictors2.items():
            if name in ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
                continue
            confidence = weights.get(name, 0.0)
            if confidence >= 0:
                try:
                    pred = model.predict_single()
                    if isinstance(pred, tuple):
                        pred = pred[1] if len(pred) > 1 else pred[0]
                    if isinstance(pred, list):
                        pred = pred[0] if pred else None
                    elif isinstance(pred, (int, float)):
                        pred = int(pred)
                    if pred is not None and 1 <= pred <= self.data.total_numbers2:
                        all_predictions2[pred] = all_predictions2.get(pred, 0) + confidence
                except Exception:
                    pass
        
        # Fallback для поля2
        if not all_predictions2:
            print("[WARN] Нет предсказаний для поля2, используем Frequency")
            from collections import Counter
            counter = Counter()
            for block in self.data.blocks2:
                if block:
                    counter.update(block)
            if counter:
                most_common = counter.most_common(1)[0][0]
                all_predictions2 = {most_common: 1.0}
            else:
                all_predictions2 = {self.data.total_numbers2 // 2: 1.0}
        
        # === ВЫБОР ТОП-N ===
        max_weight1 = max(all_predictions1.values()) if all_predictions1 else 1
        top1_preds = sorted(all_predictions1.items(), key=lambda x: x[1], reverse=True)[:n]
        max_weight2 = max(all_predictions2.values()) if all_predictions2 else 1
        top2_preds = sorted(all_predictions2.items(), key=lambda x: x[1], reverse=True)[:n]
        
        results = []
        if n == 1:
            p1 = list(top1_preds[0][0])
            p1.sort()
            p2 = [top2_preds[0][0]]
            conf = (top1_preds[0][1] / max_weight1) * 100
            if conf > 80:
                conf = 80.0
            results.append((p1, p2, round(conf, 1)))
        else:
            for i in range(n):
                if i < len(top1_preds):
                    p1 = list(top1_preds[i][0])
                else:
                    p1 = list(top1_preds[-1][0])
                p1.sort()
                if i < len(top2_preds):
                    p2 = [top2_preds[i][0]]
                else:
                    p2 = [top2_preds[-1][0]]
                conf1 = (top1_preds[min(i, len(top1_preds)-1)][1] / max_weight1) * 100
                conf2 = (top2_preds[min(i, len(top2_preds)-1)][1] / max_weight2) * 100
                conf = (conf1 + conf2) / 2
                if conf > 80:
                    conf = 80.0
                results.append((p1, p2, round(conf, 1)))
        
        return results

    def evaluate_model_on_field2(self, test_ratio: float = 0.2) -> Dict[str, float]:
        """Оценивает качество моделей для ВТОРОГО поля двойной лотереи."""
        if not self.data.is_double():
            return {}
        
        if self.data.blocks_count() < 10:
            return {}
        
        exclude = ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble")
        
        split_idx = int(self.data.blocks_count() * (1 - test_ratio))
        split_idx = max(1, min(split_idx, self.data.blocks_count() - 1))
        
        all_blocks2 = list(reversed(self.data.blocks2))
        train_blocks2 = all_blocks2[:split_idx]
        test_blocks2 = all_blocks2[split_idx:]
        
        print(f"\n[EVAL] Оценка для ПОЛЯ2: {len(train_blocks2)} train, {len(test_blocks2)} test")
        
        results = {}
        
        for name, predictor in self.predictors2.items():
            if any(e in name for e in exclude):
                continue
            
            try:
                # Обучаем
                predictor.fit(train_blocks2)
                
                total_matches = 0
                valid_predictions = 0
                
                for true_block in test_blocks2:
                    try:
                        pred = predictor.predict_single()
                        
                        # Нормализация для поля2
                        if isinstance(pred, tuple):
                            # Берём второе поле
                            pred = pred[1] if len(pred) > 1 else pred[0]
                        
                        if pred is None:
                            continue
                        
                        # Преобразуем в список
                        if not isinstance(pred, list):
                            pred = [pred]
                        
                        # Берем первое число
                        if not pred:
                            continue
                        
                        predicted_num = pred[0]
                        true_num = true_block[0] if true_block else None
                        
                        if true_num is not None and predicted_num == true_num:
                            total_matches += 1
                        
                        valid_predictions += 1
                        
                    except Exception as e:
                        continue
                
                avg_matches = total_matches / valid_predictions if valid_predictions > 0 else 0.0
                results[name] = round(avg_matches, 2)
                
            except Exception as e:
                results[name] = 0.0
        
        return results

    def evaluate_models(self, test_ratio: float = 0.2) -> Dict[str, float]:
        """
        Оценивает качество ВСЕХ моделей (включая ансамбли) на исторических данных.
        """
        if self.data.blocks_count() < 10:
            print("[WARN] Недостаточно данных для оценки (нужно минимум 10 блоков)")
            return {}
        
        # НЕ ИСКЛЮЧАЕМ АНСАМБЛИ
        # exclude = ("Ensemble", "VotingEnsemble", "WeightedEnsemble", "StackingEnsemble")
        exclude = ()
        
        # Рассчитываем индекс разделения
        split_idx = int(self.data.blocks_count() * (1 - test_ratio))
        split_idx = max(1, min(split_idx, self.data.blocks_count() - 1))
        
        # Блоки для обучения (старые) и тестирования (новые)
        all_blocks = list(reversed(self.data.blocks))
        train_blocks = all_blocks[:split_idx]
        test_blocks = all_blocks[split_idx:]
        
        print(f"\n[EVAL] Оценка {len(self.predictors1)} моделей для поля1...")
        print(f"[EVAL] Train: {len(train_blocks)} блоков, Test: {len(test_blocks)} блоков")
        
        results = {}
        evaluated = 0
        errors = 0
        
        # ==================== ОЦЕНКА ДЛЯ ПОЛЯ1 ====================
        for name, predictor in self.predictors1.items():
            # Пропускаем только если явно указано в exclude (сейчас пусто)
            if any(e in name for e in exclude):
                continue
            
            try:
                # Сохраняем исходное состояние
                was_double = getattr(predictor, 'is_double', False)
                
                # Обучаем на тренировочных данных
                if was_double and hasattr(predictor, 'fit_double'):
                    train_blocks2 = list(reversed(self.data.blocks2))[:split_idx] if self.data.blocks2 else []
                    if train_blocks2:
                        predictor.fit_double(train_blocks, train_blocks2)
                    else:
                        predictor.fit(train_blocks)
                else:
                    predictor.fit(train_blocks)
                
                total_matches = 0
                valid_predictions = 0
                
                for true_block in test_blocks:
                    try:
                        pred = predictor.predict_single()
                        
                        # Нормализация предсказания
                        if isinstance(pred, tuple):
                            pred = pred[0] if len(pred) > 0 else []
                        
                        if pred is None:
                            continue
                        
                        if not isinstance(pred, list):
                            continue
                        
                        if len(pred) != self.data.pick_count:
                            continue
                        
                        matches = len(set(pred) & set(true_block))
                        total_matches += matches
                        valid_predictions += 1
                        
                    except Exception as e:
                        continue
                
                if valid_predictions > 0:
                    avg_matches = total_matches / valid_predictions
                else:
                    avg_matches = 0.0
                
                results[name] = round(avg_matches, 2)
                evaluated += 1
                
            except Exception as e:
                print(f"  [ERR] Ошибка при оценке {name}: {e}")
                results[name] = 0.0
                errors += 1
        
        print(f"[EVAL] Поле1: оценено {evaluated} моделей, ошибок {errors}")
        
        # ==================== ОЦЕНКА ДЛЯ ПОЛЯ2 (если двойная лотерея) ====================
        if self.data.is_double():
            print(f"\n[EVAL] Оценка {len(self.predictors2)} моделей для поля2...")
            
            all_blocks2 = list(reversed(self.data.blocks2))
            train_blocks2 = all_blocks2[:split_idx]
            test_blocks2 = all_blocks2[split_idx:]
            
            evaluated2 = 0
            errors2 = 0
            
            for name, predictor in self.predictors2.items():
                if any(e in name for e in exclude):
                    continue
                
                try:
                    predictor.fit(train_blocks2)
                    
                    total_matches = 0
                    valid_predictions = 0
                    
                    for true_block in test_blocks2:
                        try:
                            pred = predictor.predict_single()
                            
                            # Нормализация для поля2
                            if isinstance(pred, tuple):
                                pred = pred[1] if len(pred) > 1 else pred[0]
                            
                            if pred is None:
                                continue
                            
                            if isinstance(pred, list):
                                pred = pred[0] if pred else None
                            elif isinstance(pred, (int, float)):
                                pred = int(pred)
                            else:
                                continue
                            
                            if not (1 <= pred <= self.data.total_numbers2):
                                continue
                            
                            true_num = true_block[0] if true_block else None
                            if true_num is not None and pred == true_num:
                                total_matches += 1
                            
                            valid_predictions += 1
                            
                        except Exception:
                            continue
                    
                    avg_matches = total_matches / valid_predictions if valid_predictions > 0 else 0.0
                    results[f"{name}_field2"] = round(avg_matches, 2)
                    evaluated2 += 1
                    
                except Exception as e:
                    print(f"  [ERR] Ошибка при оценке {name} (поле2): {e}")
                    results[f"{name}_field2"] = 0.0
                    errors2 += 1
            
            print(f"[EVAL] Поле2: оценено {evaluated2} моделей, ошибок {errors2}")
        
        return results

    def evaluate_models_detailed(self, test_ratio: float = 0.2) -> Dict[str, Dict]:
        """
        Возвращает детальную статистику для всех моделей поля1 (включая ансамбли):
        {имя_модели: {'avg': среднее_совпадений, 'avg_sq': среднее_квадратов, 'max': максимум, 'dist': распределение}}
        """
        if self.data.blocks_count() < 10:
            print("[WARN] Недостаточно данных для оценки (нужно минимум 10 блоков)")
            return {}

        split_idx = int(self.data.blocks_count() * (1 - test_ratio))
        split_idx = max(1, min(split_idx, self.data.blocks_count() - 1))

        all_blocks = list(reversed(self.data.blocks))
        train_blocks = all_blocks[:split_idx]
        test_blocks = all_blocks[split_idx:]

        print(f"\n[EVAL] Детальная оценка {len(self.predictors1)} моделей для поля1 (включая ансамбли)...")
        print(f"[EVAL] Train: {len(train_blocks)} блоков, Test: {len(test_blocks)} блоков")

        results = {}
        for name, predictor in self.predictors1.items():
            # Убираем исключение ансамблей – оцениваем все модели
            try:
                # Обучаем на тренировочных данных
                if hasattr(predictor, 'fit'):
                    predictor.fit(train_blocks)
                total_matches = 0
                total_sq = 0
                max_matches = 0
                dist = {}
                valid = 0
                for true_block in test_blocks:
                    pred = predictor.predict_single()
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if pred and len(pred) == self.data.pick_count:
                        matches = len(set(pred) & set(true_block))
                        total_matches += matches
                        total_sq += matches * matches
                        if matches > max_matches:
                            max_matches = matches
                        dist[matches] = dist.get(matches, 0) + 1
                        valid += 1
                if valid > 0:
                    avg = total_matches / valid
                    avg_sq = total_sq / valid
                else:
                    avg = 0.0
                    avg_sq = 0.0
                results[name] = {'avg': avg, 'avg_sq': avg_sq, 'max': max_matches, 'dist': dist}
            except Exception as e:
                print(f"  [ERR] Ошибка оценки {name}: {e}")
                results[name] = {'avg': 0.0, 'avg_sq': 0.0, 'max': 0, 'dist': {}}
        return results

    def _normalize_prediction(self, pred) -> List[int]:
        """
        Нормализует предсказание в единый формат списка чисел.
        
        Args:
            pred: Предсказание в любом формате
        
        Returns:
            Список чисел или None
        """
        if pred is None:
            return None
        
        # Если это кортеж (поле1, поле2) для двойной лотереи
        if isinstance(pred, tuple):
            if len(pred) >= 1:
                pred = pred[0]  # Берём первое поле
            else:
                return None
        
        # Если это число (int/float)
        if isinstance(pred, (int, float)):
            return [int(pred)]
        
        # Если это список
        if isinstance(pred, list):
            return pred
        
        # Если это numpy array
        try:
            import numpy as np
            if isinstance(pred, np.ndarray):
                return pred.tolist()
        except ImportError:
            pass
        
        return None