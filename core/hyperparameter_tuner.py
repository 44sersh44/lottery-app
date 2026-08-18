"""Walk-Forward оптимизация параметров моделей с последовательным обучением."""

import json
import os
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import itertools
from datetime import datetime
from core.lottery_data import LotteryData
from core.optimizers import OPTIMIZER_REGISTRY


class HyperparameterTuner:
    """
    Walk-Forward оптимизация параметров моделей.
    
    Алгоритм:
    1. Фиксируем начальное окно обучения (например, 80% от данных)
    2. Для каждой итерации:
       - Обучаем модели на текущем окне
       - Предсказываем следующий тираж (один из 20%)
       - Запоминаем результат
       - Расширяем окно обучения на один тираж
    3. Сохраняем все настройки и результаты
    """
    
    # Определяем параметры для перебора каждой модели
    PARAM_GRIDS = {
        # Статистические модели
        "Markov": {"order": [1, 2, 3]},
        "WeightedFreq": {"decay": [0.7, 0.8, 0.9, 0.95]},
        "Poisson": {"k": [1, 2, 3]},
        "MovingAverage": {"window": [3, 5, 7, 10]},
        "Bayesian": {"alpha": [0.3, 0.5, 0.7, 1.0]},
        "Bootstrap": {"n_resamples": [50, 100, 200]},
        "ExpSmoothing": {"alpha": [0.2, 0.3, 0.5, 0.7]},
        "GeometricMean": {},
        "HarmonicMean": {},
        "Median": {},
        
        # Классические модели
        "Frequency": {},
        "Positional": {},
        "LinearReg": {},
        "ARIMA": {"order": [(3,1,0), (5,1,0), (7,1,0), (5,1,2)]},
        
        # Машинное обучение
        "RandomForest": {"n_estimators": [50, 100, 150]},
        "XGBoost": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "BlendingEnsemble": {
            "meta_model": ["ridge", "lasso", "elasticnet", "rf"],
            "n_folds": [3, 5, 7],
            "use_original_features": [True, False]
        },
        
        # Линейные модели
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "ElasticNet": {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]},
        "Huber": {"epsilon": [1.35, 1.5, 2.0], "alpha": [0.0001, 0.001]},
        "TheilSen": {},
        "OMP": {"n_nonzero_coefs": [None, 5, 10]},
        "BayesianARD": {},
        "PassiveAggressive": {"C": [0.1, 1.0, 10.0]},
        
        # SVM/KNN
        "SVM": {"C": [0.1, 1.0, 10.0], "gamma": ['scale', 'auto']},
        "KNN": {"n_neighbors": [3, 5, 7, 10]},
        "KNNPattern": {"n_neighbors": [3, 5, 7]},
        "DBSCAN": {"eps": [0.3, 0.5, 0.7], "min_samples": [2, 3, 5]},
        "IsolationForest": {"contamination": [0.05, 0.1, 0.2]},
        
        # Ансамбли sklearn
        "ExtraTrees": {"n_estimators": [50, 100]},
        "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
        "Bagging": {"n_estimators": [10, 20, 30]},
        "HistGradientBoosting": {"max_iter": [50, 100], "learning_rate": [0.05, 0.1]},
        
        # Базовые sklearn
        "GaussianNB": {},
        "BernoulliNB": {"alpha": [0.5, 1.0, 2.0]},
        "QuantileRegressor": {"quantile": [0.3, 0.5, 0.7]},
        "RidgeCV": {"alphas": [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0]]},
        
        # Новые модели
        "LSTM": {"lookback": [3, 5, 10, 20], "epochs": [10, 20, 30]},
        "Genetic": {"population_size": [50, 100], "generations": [30, 50]},
        "Transformer": {"d_model": [32, 64], "n_heads": [4]},
        "BayesianProbability": {},
        "MonteCarlo": {"n_simulations": [200, 1000]},
        
        # Ансамбли
        "Ensemble": {
            "model_selection": ["top5", "top10", "top20", "all"],
            "strategy": ["vote", "average", "weighted"]
        },
        "VotingEnsemble": {
            "voting": ["soft", "hard"]
        },
        "WeightedEnsemble": {
            "performance_based": [True, False],
            "use_ranking": [True, False],
            "min_weight": [0.01, 0.05, 0.1]
        },
        "StackingEnsemble": {
            "use_features": [True, False],
            "cv_folds": [3, 5, 7, 10],
            "meta_model": ["ridge", "lasso", "rf", "xgboost"]
        },
    }

    # === АДАПТИВНЫЕ ШАГИ ДЛЯ РАЗНЫХ ТИПОВ ПАРАМЕТРОВ ===
    ADAPTIVE_STEPS = {
        "integer": {"coarse": 10, "medium": 5, "fine": 1, "ultra": 1},
        "float": {"coarse": 0.1, "medium": 0.05, "fine": 0.01, "ultra": 0.005},
        "tiny": {"coarse": 0.02, "medium": 0.01, "fine": 0.005, "ultra": 0.001},
        "small_int": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
    }

    PARAM_TYPES = {
        "n_estimators": "integer",
        "max_depth": "small_int",
        "window": "small_int",
        "k": "small_int",
        "order": "small_int",
        "min_samples": "small_int",
        "lookback": "small_int",
        "epochs": "small_int",
        "cv_folds": "small_int",
        "n_resamples": "integer",
        "population_size": "integer",
        "generations": "integer",
        "learning_rate": "float",
        "decay": "float",
        "alpha": "float",
        "C": "float",
        "eps": "float",
        "n_neighbors": "small_int",
        "max_iter": "small_int",
        "n_simulations": "integer",
        "contamination": "float",
        "min_weight": "tiny",
        "mutation_rate": "tiny",
    }

    PARAM_SPECIFIC_STEPS = {
        # Целочисленные параметры
        "n_estimators": {"coarse": 10, "medium": 5, "fine": 1, "ultra": 1},
        "population_size": {"coarse": 10, "medium": 5, "fine": 1, "ultra": 1},
        "generations": {"coarse": 5, "medium": 2, "fine": 1, "ultra": 1},
        "n_resamples": {"coarse": 10, "medium": 5, "fine": 1, "ultra": 1},
        "lookback": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
        "epochs": {"coarse": 5, "medium": 2, "fine": 1, "ultra": 1},
        "cv_folds": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
        # Вещественные параметры
        "learning_rate": {"coarse": 0.1, "medium": 0.05, "fine": 0.01, "ultra": 0.005},
        "decay": {"coarse": 0.05, "medium": 0.02, "fine": 0.01, "ultra": 0.005},
        "alpha": {"coarse": 0.1, "medium": 0.05, "fine": 0.01, "ultra": 0.005},
        "C": {"coarse": 0.5, "medium": 0.2, "fine": 0.1, "ultra": 0.05},
        "eps": {"coarse": 0.05, "medium": 0.02, "fine": 0.01, "ultra": 0.005},
        "contamination": {"coarse": 0.05, "medium": 0.02, "fine": 0.01, "ultra": 0.005},
        "min_weight": {"coarse": 0.02, "medium": 0.01, "fine": 0.005, "ultra": 0.001},
        "mutation_rate": {"coarse": 0.02, "medium": 0.01, "fine": 0.005, "ultra": 0.001},
        # Параметры с малыми значениями
        "order": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
        "window": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
        "k": {"coarse": 1, "medium": 1, "fine": 1, "ultra": 1},
        "min_samples": {"coarse": 1, "medium": 1, "fine": 1, "ultra": 1},
        "max_depth": {"coarse": 2, "medium": 1, "fine": 1, "ultra": 1},
    }

    PARAM_LIMITS = {
        "order": (1, 10),
        "n_estimators": (10, 1000),
        "max_depth": (1, 20),
        "window": (2, 50),
        "k": (1, 20),
        "learning_rate": (0.001, 0.5),
        "decay": (0.5, 0.99),
        "alpha": (0.001, 10.0),
        "C": (0.001, 100.0),
        "eps": (0.1, 5.0),
        "contamination": (0.01, 0.5),
        "min_weight": (0.001, 0.5),
        "mutation_rate": (0.001, 0.5),
        "population_size": (10, 500),
        "generations": (10, 200),
        "min_samples": (2, 20),
        "lookback": (2, 20),
        "quantile": (0.01, 0.99),
        "epochs": (10, 200),
        "n_resamples": (10, 1000),
        "cv_folds": (2, 10),
    }
    
    def __init__(self, data: LotteryData, profile_path: str):
        self.data = data
        self.profile_path = profile_path
        self.results_file = os.path.join(profile_path, "walk_forward_results.json")
        self.iterations_log = os.path.join(profile_path, "iterations_log.json")
        # Регистрируем оптимизаторы
        self.optimizers = {name: cls(self) for name, cls in OPTIMIZER_REGISTRY.items()}
    
    def get_param_grid_by_step(self, step: str = "medium") -> Dict:
        """Возвращает сетку параметров в зависимости от выбранного шага."""
        base_grid = self.PARAM_GRIDS.copy()
        
        if step == "fine":
            # Расширяем сетку параметров
            base_grid["Markov"] = {"order": [1, 2, 3, 4, 5]}
            base_grid["WeightedFreq"] = {"decay": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}
            base_grid["Poisson"] = {"k": [1, 2, 3, 4, 5]}
            base_grid["MovingAverage"] = {"window": [3, 4, 5, 6, 7, 8, 9, 10]}
            base_grid["RandomForest"] = {"n_estimators": [50, 75, 100, 125, 150]}
            base_grid["XGBoost"] = {
                "n_estimators": [50, 75, 100],
                "learning_rate": [0.05, 0.075, 0.1, 0.15, 0.2],
                "max_depth": [3, 4, 5, 6, 7]
            }
            base_grid["KNN"] = {"n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10]}
            base_grid["DBSCAN"] = {"eps": [0.3, 0.4, 0.5, 0.6, 0.7], "min_samples": [2, 3, 4, 5]}
            
            # Ансамбли для fine шага
            base_grid["Ensemble"] = {
                "model_selection": ["top5", "top10", "top20", "all"],
                "strategy": ["vote", "average", "weighted"]
            }
            base_grid["VotingEnsemble"] = {"voting": ["soft", "hard"]}
            base_grid["WeightedEnsemble"] = {
                "performance_based": [True, False],
                "use_ranking": [True, False],
                "min_weight": [0.005, 0.01, 0.05, 0.1]
            }
            base_grid["StackingEnsemble"] = {
                "use_features": [True, False],
                "cv_folds": [3, 5, 7, 10],
                "meta_model": ["ridge", "lasso", "rf", "xgboost"]
            }
            
            # Линейные модели для fine шага
            base_grid["Ridge"] = {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}
            base_grid["Lasso"] = {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
            base_grid["ElasticNet"] = {"alpha": [0.1, 0.5, 1.0], "l1_ratio": [0.3, 0.5, 0.7, 0.9]}
            base_grid["Huber"] = {"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [0.0001, 0.001, 0.01]}
            
            # SVM для fine шага
            base_grid["SVM"] = {"C": [0.1, 0.5, 1.0, 5.0, 10.0], "gamma": ["scale", "auto", 0.1, 0.5, 1.0]}
            base_grid["KNNPattern"] = {"n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10]}
            
            # Ансамбли sklearn для fine шага
            base_grid["ExtraTrees"] = {"n_estimators": [50, 75, 100, 150]}
            base_grid["GradientBoosting"] = {"n_estimators": [50, 75, 100], "learning_rate": [0.05, 0.1, 0.15]}
            base_grid["AdaBoost"] = {"n_estimators": [50, 75, 100], "learning_rate": [0.5, 0.75, 1.0]}
            base_grid["Bagging"] = {"n_estimators": [10, 15, 20, 25, 30]}
            base_grid["HistGradientBoosting"] = {"max_iter": [50, 75, 100], "learning_rate": [0.05, 0.1, 0.15]}
            
            # Базовые sklearn для fine шага
            base_grid["BernoulliNB"] = {"alpha": [0.5, 0.7, 1.0, 1.5, 2.0]}
            base_grid["QuantileRegressor"] = {"quantile": [0.3, 0.4, 0.5, 0.6, 0.7]}
            
            base_grid["Bayesian"] = {"alpha": [0.1, 0.3, 0.5, 0.7, 1.0]}
            base_grid["Bootstrap"] = {"n_resamples": [50, 100, 150, 200]}
            base_grid["ExpSmoothing"] = {"alpha": [0.1, 0.2, 0.3, 0.5, 0.7]}
            base_grid["IsolationForest"] = {"contamination": [0.05, 0.1, 0.15, 0.2]}
        
        elif step == "coarse":
            # Уменьшаем сетку параметров
            base_grid["Markov"] = {"order": [1, 3, 5]}
            base_grid["WeightedFreq"] = {"decay": [0.7, 0.85, 0.95]}
            base_grid["Poisson"] = {"k": [1, 3, 5]}
            base_grid["MovingAverage"] = {"window": [3, 7, 10]}
            base_grid["RandomForest"] = {"n_estimators": [50, 150]}
            base_grid["XGBoost"] = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.2],
                "max_depth": [3, 7]
            }
            base_grid["KNN"] = {"n_neighbors": [3, 7, 10]}
            base_grid["DBSCAN"] = {"eps": [0.3, 0.7], "min_samples": [2, 5]}
            
            # Ансамбли для coarse шага
            base_grid["Ensemble"] = {
                "model_selection": ["top10", "top20"],
                "strategy": ["vote", "weighted"]
            }
            base_grid["VotingEnsemble"] = {"voting": ["soft"]}
            base_grid["WeightedEnsemble"] = {"performance_based": [True], "use_ranking": [True, False], "min_weight": [0.01, 0.05]}
            base_grid["StackingEnsemble"] = {
                "use_features": [True],
                "cv_folds": [5],
                "meta_model": ["ridge", "rf"]
            }
            
            # Линейные модели для coarse шага
            base_grid["Ridge"] = {"alpha": [0.1, 1.0, 10.0]}
            base_grid["Lasso"] = {"alpha": [0.01, 0.1, 1.0]}
            base_grid["ElasticNet"] = {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]}
            base_grid["Huber"] = {"epsilon": [1.35, 1.5], "alpha": [0.0001, 0.001]}
            
            # SVM для coarse шага
            base_grid["SVM"] = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
            base_grid["KNNPattern"] = {"n_neighbors": [3, 5, 7]}
            
            # Ансамбли sklearn для coarse шага
            base_grid["ExtraTrees"] = {"n_estimators": [50, 100]}
            base_grid["GradientBoosting"] = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            base_grid["AdaBoost"] = {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
            base_grid["Bagging"] = {"n_estimators": [10, 20, 30]}
            base_grid["HistGradientBoosting"] = {"max_iter": [50, 100], "learning_rate": [0.05, 0.1]}
            
            # Базовые sklearn для coarse шага
            base_grid["BernoulliNB"] = {"alpha": [0.5, 1.0, 2.0]}
            base_grid["QuantileRegressor"] = {"quantile": [0.3, 0.5, 0.7]}
            
            base_grid["Bayesian"] = {"alpha": [0.3, 0.5, 1.0]}
            base_grid["Bootstrap"] = {"n_resamples": [100, 200]}
            base_grid["ExpSmoothing"] = {"alpha": [0.2, 0.3, 0.5]}
            base_grid["IsolationForest"] = {"contamination": [0.05, 0.1]}
        
        return base_grid

    def _get_current_combination(self, model_name: str, field: int, param_names: List[str]) -> Dict[str, Any]:
        try:
            from core.parameter_manager import ParameterManager
            pm = ParameterManager(self.profile_path)
            if field == 2 and model_name not in ('Field2Wrapper', 'CompositePredictor'):
                full_name = f"{model_name}_field2"
            else:
                full_name = model_name
            
            current_params = {}
            for param_name in param_names:
                val = pm.get_param(full_name, param_name, None)
                if val is not None:
                    current_params[param_name] = val
            return current_params if current_params else None
        except Exception as e:
            print(f"  [WARN] Не удалось загрузить текущие параметры для {model_name}: {e}")
            return None

    def evaluate_combination(self, model_name: str, params: Dict[str, Any], field: int = 1, test_ratio: float = 0.2) -> float:
        if field == 1:
            all_blocks = list(reversed(self.data.blocks))
            pick_count = self.data.pick_count
            total_numbers = self.data.total_numbers
        else:
            all_blocks = list(reversed(self.data.blocks2))
            pick_count = self.data.pick_count2
            total_numbers = self.data.total_numbers2

        total_blocks = len(all_blocks)
        if total_blocks < 10:
            return 0.0

        split_idx = int(total_blocks * (1 - test_ratio))
        train_blocks = all_blocks[:split_idx]
        test_blocks = all_blocks[split_idx:]

        model = self._create_model(model_name, total_numbers, pick_count, params)
        if model is None:
            return 0.0

        try:
            model.fit(train_blocks)
            total_matches = 0
            valid = 0
            for true_block in test_blocks:
                pred = model.predict_single()
                if field == 1:
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if pred and len(pred) == pick_count:
                        matches = len(set(pred) & set(true_block))
                        total_matches += matches
                        valid += 1
                else:
                    if isinstance(pred, tuple):
                        pred = pred[1] if len(pred) > 1 else pred[0]
                    if isinstance(pred, list):
                        pred = pred[0] if pred else None
                    if pred is not None and pred == true_block[0]:
                        total_matches += 1
                        valid += 1
            avg = total_matches / valid if valid > 0 else 0.0
            return avg
        except Exception:
            return 0.0

    def _generate_adaptive_combinations(self, param_names: List[str], param_grid: Dict[str, List],
                                        current_params: Dict[str, Any], level: str,
                                        param_ranges: Dict[str, tuple]) -> List[tuple]:
        import math
        import itertools

        steps = {}
        for p in param_names:
            if p in param_grid and param_grid[p] and not all(isinstance(x, (int, float)) for x in param_grid[p]):
                steps[p] = None
            else:
                if p in self.PARAM_SPECIFIC_STEPS and level in self.PARAM_SPECIFIC_STEPS[p]:
                    step = self.PARAM_SPECIFIC_STEPS[p][level]
                else:
                    param_type = self.PARAM_TYPES.get(p, "integer")
                    step = self.ADAPTIVE_STEPS.get(param_type, {}).get(level, 1)
                steps[p] = step

        max_combinations = 500
        attempts = 0
        while attempts < 5:
            values_per_param = []
            total_combos = 1
            for p in param_names:
                if steps[p] is None:
                    vals = param_grid[p]
                    values_per_param.append(vals)
                    total_combos *= len(vals)
                else:
                    min_val, max_val = param_ranges.get(p, (None, None))
                    if min_val is None or max_val is None:
                        current = current_params.get(p, 0)
                        vals = [current]
                    else:
                        step = steps[p]
                        vals = []
                        val = min_val
                        while val <= max_val + 1e-9:
                            if isinstance(current_params.get(p, 0), int):
                                val_int = int(round(val))
                                vals.append(val_int)
                            else:
                                precision = int(round(-math.log10(step))) if step < 1 else 0
                                vals.append(round(val, precision))
                            val += step
                        vals = sorted(set(vals))
                    values_per_param.append(vals)
                    total_combos *= len(vals)
            if total_combos <= max_combinations:
                break
            for p in param_names:
                if steps[p] is not None:
                    steps[p] *= 2
            attempts += 1

        if attempts == 5:
            for i, p in enumerate(param_names):
                if steps[p] is not None and len(values_per_param[i]) > 20:
                    step_idx = len(values_per_param[i]) // 20
                    values_per_param[i] = values_per_param[i][::max(1, step_idx)][:20]

        combos = list(itertools.product(*values_per_param))
        return combos

    def run_tuning(self, field: int = 1, initial_ratio: float = 0.8, step: str = "medium",
                   model_names: List[str] = None, output_file: str = None,
                   use_current_params: bool = False, adaptive: bool = False) -> Dict[str, Any]:
        import gc
        import json
        import os
        import itertools

        print("\n" + "=" * 80)
        print(f"🔧 WALK-FORWARD ОПТИМИЗАЦИЯ — ПОЛЕ {field}")
        print("=" * 80)
        print(f"📊 Выбранный метод: {step}")

        param_grids = self.get_param_grid_by_step(step if step not in ["bayesian", "hybrid", "hyperband", "genetic", "bohb", "auto_heavy"] else "medium")

        if field == 1:
            all_blocks = list(reversed(self.data.blocks))
            pick_count = self.data.pick_count
            total_numbers = self.data.total_numbers
        else:
            all_blocks = list(reversed(self.data.blocks2))
            pick_count = self.data.pick_count2
            total_numbers = self.data.total_numbers2

        total_blocks = len(all_blocks)
        if total_blocks < 10:
            print(f"[ERR] Недостаточно данных: {total_blocks} блоков (нужно минимум 10)")
            return {}

        initial_window = int(total_blocks * initial_ratio)
        initial_window = max(5, min(initial_window, total_blocks - 2))

        print(f"\n[DATA] Всего блоков: {total_blocks}")
        print(f"[DATA] Начальное окно: {initial_window} блоков")
        print(f"[DATA] Количество итераций: {total_blocks - initial_window}")

        all_results = {}

        if model_names is not None:
            param_grids = {name: grid for name, grid in param_grids.items() if name in model_names}
            if not param_grids:
                print(f"[ERR] Нет моделей для оптимизации из списка: {model_names}")
                return {}

        # БАЙЕСОВСКИЙ
        if step == "bayesian":
            print("\n[BAYESIAN] Включена байесовская оптимизация (Optuna)")
            n_trials = 50
            print(f"   Количество проб: {n_trials}")
            optimizer = self.optimizers['bayesian']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                try:
                    result = optimizer.optimize(
                        model_name, param_grid, all_blocks, initial_window,
                        pick_count, total_numbers, field, n_trials=n_trials
                    )
                except Exception as e:
                    print(f"  [ERR] Ошибка байесовской оптимизации: {e}")
                    print("  [WARN] Используем обычный перебор для этой модели")
                    continue
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (байесовские): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (байесовские): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # ГИБРИДНЫЙ
        if step == "hybrid":
            print("\n[HYBRID] Включён гибридный режим: адаптивный coarse + байесовское уточнение")
            print(f"   Количество проб на втором этапе: 30")
            optimizer = self.optimizers['hybrid']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 ГИБРИДНАЯ ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                try:
                    result = optimizer.optimize(
                        model_name, param_grid, all_blocks, initial_window,
                        pick_count, total_numbers, field,
                        start_params={}, n_trials=30
                    )
                except Exception as e:
                    print(f"  [ERR] Ошибка гибридной оптимизации: {e}")
                    continue
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (гибридные): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (гибридные): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # HYPERBAND
        if step == "hyperband":
            print("\n[HYPERBAND] Включён Hyperband (быстрый поиск с ранней остановкой)")
            optimizer = self.optimizers['hyperband']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 HYPERBAND ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                try:
                    result = optimizer.optimize(
                        model_name, param_grid, all_blocks, initial_window,
                        pick_count, total_numbers, field,
                        max_iter=100, eta=3
                    )
                except Exception as e:
                    print(f"  [ERR] Ошибка Hyperband: {e}")
                    continue
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (Hyperband): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (Hyperband): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # ГЕНЕТИЧЕСКИЙ
        if step == "genetic":
            print("\n[GENETIC] Включён генетический алгоритм")
            optimizer = self.optimizers['genetic']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 ГЕНЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                try:
                    result = optimizer.optimize(
                        model_name, param_grid, all_blocks, initial_window,
                        pick_count, total_numbers, field,
                        population_size=20, generations=10, mutation_rate=0.1
                    )
                except Exception as e:
                    print(f"  [ERR] Ошибка генетического алгоритма: {e}")
                    continue
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (Genetic): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (Genetic): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # BOHB
        if step == "bohb":
            print("\n[BOHB] Включён BOHB (Bayesian Optimization + Hyperband)")
            optimizer = self.optimizers['bohb']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 BOHB ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                try:
                    result = optimizer.optimize(
                        model_name, param_grid, all_blocks, initial_window,
                        pick_count, total_numbers, field, n_trials=50
                    )
                except Exception as e:
                    print(f"  [ERR] Ошибка BOHB: {e}")
                    print("  [WARN] Переключение на Hyperband")
                    try:
                        fallback = self.optimizers['hyperband']
                        result = fallback.optimize(
                            model_name, param_grid, all_blocks, initial_window,
                            pick_count, total_numbers, field, max_iter=100, eta=3
                        )
                    except Exception as e2:
                        print(f"  [ERR] Ошибка Hyperband: {e2}")
                        continue
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (BOHB): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (BOHB): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # АВТОМАТИЧЕСКИЙ
        if step == "auto_heavy":
            print("\n[AUTO] Автоматический выбор метода оптимизации:")
            print("  - ARIMA → быстрый перебор (coarse)")
            print("  - LSTM, Transformer → гибридный (coarse + байесовский) с малым числом проб")
            print("  - Prophet → Hyperband (уменьшенный бюджет)")
            print("  - Остальные → гибридный (coarse + байесовский)")

            arima_models = ["ARIMA"]
            lstm_models = ["LSTM", "Transformer"]
            prophet_models = ["Prophet"]

            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")

                try:
                    if model_name in arima_models:
                        print(f"  [AUTO] Используем быстрый перебор coarse для ARIMA")
                        optimizer = self.optimizers['grid_coarse']
                        result = optimizer.optimize(
                            model_name, param_grid, all_blocks, initial_window,
                            pick_count, total_numbers, field
                        )
                    elif model_name in lstm_models:
                        print(f"  [AUTO] Используем гибридный метод для {model_name} (медленная нейросеть)")
                        optimizer = self.optimizers['hybrid']
                        result = optimizer.optimize(
                            model_name, param_grid, all_blocks, initial_window,
                            pick_count, total_numbers, field,
                            start_params={}, n_trials=15
                        )
                    elif model_name in prophet_models:
                        print(f"  [AUTO] Используем Hyperband для Prophet (уменьшенный бюджет)")
                        optimizer = self.optimizers['hyperband']
                        result = optimizer.optimize(
                            model_name, param_grid, all_blocks, initial_window,
                            pick_count, total_numbers, field,
                            max_iter=20, eta=2
                        )
                    else:
                        print(f"  [AUTO] Используем гибридный метод для {model_name} (обычная модель)")
                        optimizer = self.optimizers['hybrid']
                        result = optimizer.optimize(
                            model_name, param_grid, all_blocks, initial_window,
                            pick_count, total_numbers, field,
                            start_params={}, n_trials=30
                        )
                except Exception as e:
                    print(f"  [ERR] Ошибка оптимизации: {e}")
                    continue

                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result

                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ: {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window})")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ: {result['best_params']} → {result['best_score']:.3f}")

            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # АДАПТИВНЫЙ
        if adaptive:
            print("\n[ADAPTIVE] Включён адаптивный многоуровневый поиск (coarse → medium → fine → ultra)")
            optimizer = self.optimizers['adaptive']
            for model_name, param_grid in param_grids.items():
                print(f"\n{'='*60}")
                print(f"📊 АДАПТИВНАЯ ОПТИМИЗАЦИЯ: {model_name}")
                print(f"{'='*60}")
                if use_current_params:
                    from core.parameter_manager import ParameterManager
                    pm = ParameterManager(self.profile_path)
                    start_params = {}
                    for p in param_grid.keys():
                        val = pm.get_param(model_name, p, None)
                        if val is not None:
                            start_params[p] = val
                    if not start_params:
                        start_params = {p: param_grid[p][len(param_grid[p])//2] for p in param_grid if param_grid[p]}
                else:
                    start_params = {p: param_grid[p][len(param_grid[p])//2] for p in param_grid if param_grid[p]}
                result = optimizer.optimize(
                    model_name, param_grid, all_blocks, initial_window,
                    pick_count, total_numbers, field,
                    start_params=start_params
                )
                if result["best_params"]:
                    cleaned = {}
                    for k, v in result["best_params"].items():
                        if v is None:
                            continue
                        if model_name == "Markov" and k == "order" and isinstance(v, int):
                            v = (v, 1, 0)
                        cleaned[k] = v
                    result["best_params"] = cleaned
                all_results[model_name] = result
                if field == 1:
                    correct_count = int(result["best_score"] * (total_blocks - initial_window))
                    print(f"\n  🏆 ЛУЧШИЕ (адаптивные): {result['best_params']} → угадал: {result['best_score']:.0%} ({correct_count}/{total_blocks - initial_window}), ср.совпадений: {result['best_avg_matches']:.2f}")
                else:
                    print(f"\n  🏆 ЛУЧШИЕ (адаптивные): {result['best_params']} → {result['best_score']:.3f}")
            if output_file is not None:
                self._save_results(all_results, field, initial_window, output_file)
            else:
                self._save_results(all_results, field, initial_window)
            self._analyze_results(all_results, field)
            self._print_recommendations(all_results, field)
            return all_results

        # ОБЫЧНЫЙ (СЕТКА)
        for model_name, param_grid in param_grids.items():
            print(f"\n{'='*60}")
            print(f"📊 ОПТИМИЗАЦИЯ: {model_name}")
            print(f"{'='*60}")

            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            if param_values:
                combinations = list(itertools.product(*param_values))
            else:
                combinations = [()]

            if use_current_params:
                current_combo = self._get_current_combination(model_name, field, param_names)
                if current_combo is not None:
                    combo_tuple = tuple(current_combo.get(p, None) for p in param_names)
                    if combo_tuple not in combinations:
                        combinations = [combo_tuple] + combinations
                        print(f"  [INFO] Добавлена текущая комбинация параметров: {current_combo}")
                    else:
                        print(f"  [INFO] Текущая комбинация уже есть в сетке")

            if step == "fine":
                max_combinations = 200
            elif step == "coarse":
                max_combinations = 20
            else:
                max_combinations = 60

            if len(combinations) > max_combinations:
                combinations = combinations[:max_combinations]

            print(f"  Параметров для проверки: {len(combinations)}")

            best_params = None
            best_total_score = -1
            best_avg_matches = 0
            best_matches_list = []
            best_iterations_file = None

            for combo in combinations:
                params = dict(zip(param_names, combo)) if param_names else {}
                score, iterations, avg_matches, total_matches, matches_list = self._walk_forward_test(
                    model_name, params, all_blocks, initial_window,
                    pick_count, total_numbers, field
                )
                total_correct = sum(1 for it in iterations if it.get("correct", False))
                total_tests = len(iterations)
                status = "✅" if score > best_total_score else "  "
                if field == 1:
                    print(f"  {status} {params} → угадал: {score:.0%} ({total_correct}/{total_tests}), ср.совпадений: {avg_matches:.2f}")
                else:
                    print(f"  {status} {params} → {score:.3f} ({total_correct}/{total_tests})")
                if score > best_total_score:
                    best_total_score = score
                    best_params = params
                    best_avg_matches = avg_matches
                    best_matches_list = matches_list
                    iter_filename = f"wf_iter_{model_name}_field{field}.json"
                    best_iterations_file = os.path.join(self.profile_path, iter_filename)
                    with open(best_iterations_file, 'w', encoding='utf-8') as f:
                        json.dump(iterations, f, indent=2, ensure_ascii=False)
                del iterations
                if len(combinations) > 10 and (combinations.index(combo) % 10 == 0):
                    gc.collect()

            if best_params:
                cleaned_params = {}
                for k, v in best_params.items():
                    if v is None:
                        print(f"  [WARN] Параметр {k}=None пропущен")
                        continue
                    if model_name == "Markov" and k == "order" and isinstance(v, int):
                        v = (v, 1, 0)
                        print(f"  [FIX] Markov.order преобразован в {v}")
                    cleaned_params[k] = v
                best_params = cleaned_params

            all_results[model_name] = {
                "best_params": best_params,
                "best_score": best_total_score,
                "best_avg_matches": best_avg_matches,
                "best_total_matches": sum(best_matches_list),
                "matches_list": best_matches_list,
                "total_iterations": total_blocks - initial_window,
                "iterations_file": best_iterations_file
            }
            del best_matches_list
            gc.collect()

        if output_file is not None:
            self._save_results(all_results, field, initial_window, output_file)
        else:
            self._save_results(all_results, field, initial_window)

        self._analyze_results(all_results, field)
        self._print_recommendations(all_results, field)

        return all_results

    # ---------- ОБЩИЕ МЕТОДЫ ----------

    def _walk_forward_test(self, model_name, params, all_blocks, initial_window, pick_count, total_numbers, field, max_iterations=None):
        if field == 2 and pick_count == 1:
            unsuitable_models = ["Markov", "Positional", "ARIMA", "MovingAverage", 
                                  "ExpSmoothing", "GeometricMean", "HarmonicMean", "Median"]
            if model_name in unsuitable_models:
                return 0, [], 0, 0, []
        
        train_size = initial_window
        total_correct = 0
        total_tests = 0
        iterations = []
        total_matches = 0
        matches_list = []
        
        while train_size < len(all_blocks):
            if max_iterations is not None and total_tests >= max_iterations:
                break
            
            train_blocks = all_blocks[:train_size]
            test_block = all_blocks[train_size]
            
            model = self._create_model(model_name, total_numbers, pick_count, params)
            if model is None:
                break
            
            try:
                model.fit(train_blocks)
                pred = model.predict_single()
                
                correct = False
                matches_count = 0
                
                if field == 1:
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if pred and len(pred) == pick_count:
                        matches_count = len(set(pred) & set(test_block))
                        total_matches += matches_count
                        matches_list.append(matches_count)
                        if matches_count > 0:
                            correct = True
                            total_correct += 1
                else:
                    if isinstance(pred, tuple):
                        pred = pred[1] if len(pred) > 1 else pred[0]
                    if isinstance(pred, list):
                        pred = pred[0] if pred else None
                    true_num = test_block[0] if test_block else None
                    if pred == true_num:
                        correct = True
                        total_correct += 1
                        matches_count = 1
                    total_matches += matches_count
                    matches_list.append(matches_count)
                
                iterations.append({
                    "iteration": train_size + 1,
                    "correct": correct,
                    "matches": matches_count,
                    "prediction": str(pred),
                    "true_block": test_block
                })
                total_tests += 1
                
            except Exception as e:
                iterations.append({
                    "iteration": train_size + 1,
                    "correct": False,
                    "matches": 0,
                    "error": str(e)
                })
                matches_list.append(0)
                total_tests += 1
            
            train_size += 1
        
        score = total_correct / total_tests if total_tests > 0 else 0
        avg_matches = total_matches / total_tests if total_tests > 0 else 0
        
        return score, iterations, avg_matches, total_matches, matches_list

    def _create_model(self, name: str, total: int, pick: int, params: Dict):
        try:
            clean_params = {k: v for k, v in params.items() if v is not None}

            if name == "Markov":
                from predictors.markov import MarkovPredictor
                order = clean_params.get("order", 1)
                if isinstance(order, (float, tuple, list)):
                    if isinstance(order, (tuple, list)):
                        order = order[0] if order else 1
                    else:
                        order = int(round(order))
                if order < 1:
                    order = 1
                return MarkovPredictor(total, pick, order=order)

            elif name == "WeightedFreq":
                from predictors.weighted_freq import WeightedFrequencyPredictor
                decay = clean_params.get("decay", 0.9)
                return WeightedFrequencyPredictor(total, pick, decay=decay)

            elif name == "Poisson":
                from predictors.poisson import PoissonPredictor
                k = clean_params.get("k", 1)
                if isinstance(k, float):
                    k = int(round(k))
                if k < 1:
                    k = 1
                return PoissonPredictor(total, pick, k=k)

            elif name == "MovingAverage":
                from predictors.statistical import MovingAveragePredictor
                window = clean_params.get("window", 5)
                if isinstance(window, float):
                    window = int(round(window))
                if window < 1:
                    window = 1
                return MovingAveragePredictor(total, pick, window=window)

            elif name == "Bayesian":
                from predictors.statistical import BayesianPredictor
                alpha = clean_params.get("alpha", 0.5)
                return BayesianPredictor(total, pick, alpha=alpha)

            elif name == "Bootstrap":
                from predictors.statistical import BootstrapPredictor
                n_res = clean_params.get("n_resamples", 100)
                if isinstance(n_res, float):
                    n_res = int(round(n_res))
                if n_res < 1:
                    n_res = 1
                return BootstrapPredictor(total, pick, n_resamples=n_res)

            elif name == "ExpSmoothing":
                from predictors.statistical import ExponentialSmoothingPredictor
                alpha = clean_params.get("alpha", 0.3)
                return ExponentialSmoothingPredictor(total, pick, alpha=alpha)

            elif name in ["GeometricMean", "HarmonicMean", "Median"]:
                from predictors.statistical import GeometricMeanPredictor, HarmonicMeanPredictor, MedianPredictor
                if name == "GeometricMean":
                    return GeometricMeanPredictor(total, pick)
                elif name == "HarmonicMean":
                    return HarmonicMeanPredictor(total, pick)
                else:
                    return MedianPredictor(total, pick)

            elif name == "Frequency":
                from predictors.frequency import FrequencyPredictor
                return FrequencyPredictor(total, pick)

            elif name == "Positional":
                from predictors.positional import PositionalPredictor
                return PositionalPredictor(total, pick)

            elif name == "LinearReg":
                from predictors.linear_regression import LinearRegressionPredictor
                return LinearRegressionPredictor(total, pick)

            elif name == "ARIMA":
                from predictors.arima import ArimaPredictor
                order = clean_params.get("order", (5, 1, 0))
                return ArimaPredictor(total, pick, order=order)

            elif name == "RandomForest":
                from predictors.random_forest import RandomForestPredictor
                n_est = clean_params.get("n_estimators", 50)
                if isinstance(n_est, float):
                    n_est = int(round(n_est))
                if n_est < 1:
                    n_est = 1
                return RandomForestPredictor(total, pick, n_estimators=n_est)

            elif name == "XGBoost":
                from predictors.xgboost import XGBoostPredictor
                n_est = clean_params.get("n_estimators", 100)
                lr = clean_params.get("learning_rate", 0.1)
                depth = clean_params.get("max_depth", 3)
                if isinstance(n_est, float):
                    n_est = int(round(n_est))
                if isinstance(depth, float):
                    depth = int(round(depth))
                if n_est < 1:
                    n_est = 1
                if depth < 1:
                    depth = 1
                return XGBoostPredictor(total, pick, n_estimators=n_est, learning_rate=lr, max_depth=depth)

            elif name == "Ridge":
                from predictors.linear_models import RidgePredictor
                alpha = clean_params.get("alpha", 1.0)
                return RidgePredictor(total, pick, alpha=alpha)

            elif name == "Lasso":
                from predictors.linear_models import LassoPredictor
                alpha = clean_params.get("alpha", 1.0)
                return LassoPredictor(total, pick, alpha=alpha)

            elif name == "ElasticNet":
                from predictors.linear_models import ElasticNetPredictor
                alpha = clean_params.get("alpha", 1.0)
                l1 = clean_params.get("l1_ratio", 0.5)
                return ElasticNetPredictor(total, pick, alpha=alpha, l1_ratio=l1)

            elif name == "Huber":
                from predictors.linear_models import HuberPredictor
                eps = clean_params.get("epsilon", 1.35)
                alpha = clean_params.get("alpha", 0.0001)
                return HuberPredictor(total, pick, epsilon=eps, alpha=alpha)

            elif name == "TheilSen":
                from predictors.linear_models import TheilSenPredictor
                return TheilSenPredictor(total, pick)

            elif name == "OMP":
                from predictors.linear_models import OMPPredictor
                nz = clean_params.get("n_nonzero_coefs", None)
                return OMPPredictor(total, pick, n_nonzero_coefs=nz)

            elif name == "BayesianARD":
                from predictors.linear_models import BayesianARDPredictor
                return BayesianARDPredictor(total, pick)

            elif name == "PassiveAggressive":
                from predictors.linear_models import PassiveAggressivePredictor
                C = clean_params.get("C", 1.0)
                return PassiveAggressivePredictor(total, pick, C=C)

            elif name == "SVM":
                from predictors.svm_knn import SVMPredictor
                return SVMPredictor(total, pick)

            elif name == "KNN":
                from predictors.svm_knn import KNNPredictor
                k = clean_params.get("n_neighbors", 5)
                if isinstance(k, float):
                    k = int(round(k))
                if k < 1:
                    k = 1
                return KNNPredictor(total, pick, n_neighbors=k)

            elif name == "KNNPattern":
                from predictors.svm_knn import KNNPatternPredictor
                k = clean_params.get("n_neighbors", 3)
                if isinstance(k, float):
                    k = int(round(k))
                if k < 1:
                    k = 1
                return KNNPatternPredictor(total, pick, n_neighbors=k)

            elif name == "DBSCAN":
                from predictors.svm_knn import DBSCANPredictor
                eps = clean_params.get("eps", 0.5)
                min_s = clean_params.get("min_samples", 2)
                if isinstance(min_s, float):
                    min_s = int(round(min_s))
                if min_s < 1:
                    min_s = 1
                return DBSCANPredictor(total, pick, eps=eps, min_samples=min_s)

            elif name == "IsolationForest":
                from predictors.svm_knn import IsolationForestPredictor
                cont = clean_params.get("contamination", 0.1)
                return IsolationForestPredictor(total, pick, contamination=cont)

            elif name == "ExtraTrees":
                from predictors.sklearn_ensemble import ExtraTreesPredictor
                n = clean_params.get("n_estimators", 100)
                if isinstance(n, float):
                    n = int(round(n))
                if n < 1:
                    n = 1
                return ExtraTreesPredictor(total, pick, n_estimators=n)

            elif name == "GradientBoosting":
                from predictors.sklearn_ensemble import GradientBoostingPredictor
                n = clean_params.get("n_estimators", 100)
                lr = clean_params.get("learning_rate", 0.1)
                if isinstance(n, float):
                    n = int(round(n))
                if n < 1:
                    n = 1
                return GradientBoostingPredictor(total, pick, n_estimators=n, learning_rate=lr)

            elif name == "AdaBoost":
                from predictors.sklearn_ensemble import AdaBoostPredictor
                n = clean_params.get("n_estimators", 50)
                lr = clean_params.get("learning_rate", 1.0)
                if isinstance(n, float):
                    n = int(round(n))
                if n < 1:
                    n = 1
                return AdaBoostPredictor(total, pick, n_estimators=n, learning_rate=lr)

            elif name == "Bagging":
                from predictors.sklearn_ensemble import BaggingPredictor
                n = clean_params.get("n_estimators", 10)
                if isinstance(n, float):
                    n = int(round(n))
                if n < 1:
                    n = 1
                return BaggingPredictor(total, pick, n_estimators=n)

            elif name == "HistGradientBoosting":
                from predictors.sklearn_ensemble import HistGradientBoostingPredictor
                n = clean_params.get("max_iter", 100)
                lr = clean_params.get("learning_rate", 0.1)
                if isinstance(n, float):
                    n = int(round(n))
                if n < 1:
                    n = 1
                return HistGradientBoostingPredictor(total, pick, max_iter=n, learning_rate=lr)

            elif name == "GaussianNB":
                from predictors.sklearn_basic import GaussianNBPredictor
                return GaussianNBPredictor(total, pick)

            elif name == "BernoulliNB":
                from predictors.sklearn_basic import BernoulliNBPredictor
                return BernoulliNBPredictor(total, pick)

            elif name == "QuantileRegressor":
                from predictors.sklearn_basic import QuantileRegressorPredictor
                q = clean_params.get("quantile", 0.5)
                if q is None or q <= 0.0 or q >= 1.0:
                    q = 0.5
                return QuantileRegressorPredictor(total, pick, quantile=q)

            elif name == "RidgeCV":
                from predictors.sklearn_basic import RidgeCVPredictor
                alphas = clean_params.get("alphas", [0.1, 1.0, 10.0])
                return RidgeCVPredictor(total, pick, alphas=alphas)

            elif name == "LSTM":
                try:
                    from predictors.neural.lstm_predictor import LSTMPredictor
                    lookback = clean_params.get("lookback", 5)
                    epochs = clean_params.get("epochs", 20)
                    if isinstance(lookback, float):
                        lookback = int(round(lookback))
                    if isinstance(epochs, float):
                        epochs = int(round(epochs))
                    if lookback < 1:
                        lookback = 1
                    if epochs < 1:
                        epochs = 1
                    return LSTMPredictor(total, pick, lookback=lookback, epochs=epochs)
                except ImportError:
                    from predictors.frequency import FrequencyPredictor
                    return FrequencyPredictor(total, pick)

            elif name == "Genetic":
                try:
                    from predictors.genetic.genetic_predictor import GeneticPredictor
                    pop_size = clean_params.get("population_size", 50)
                    generations = clean_params.get("generations", 30)
                    if isinstance(pop_size, float):
                        pop_size = int(round(pop_size))
                    if isinstance(generations, float):
                        generations = int(round(generations))
                    if pop_size < 1:
                        pop_size = 1
                    if generations < 1:
                        generations = 1
                    return GeneticPredictor(total, pick, population_size=pop_size, generations=generations)
                except ImportError:
                    from predictors.frequency import FrequencyPredictor
                    return FrequencyPredictor(total, pick)

            elif name == "Prophet":
                try:
                    from predictors.neural.prophet_predictor import ProphetPredictor
                    seasonality = clean_params.get("seasonality_mode", "additive")
                    return ProphetPredictor(total, pick, seasonality_mode=seasonality)
                except ImportError:
                    from predictors.frequency import FrequencyPredictor
                    return FrequencyPredictor(total, pick)

            elif name == "Transformer":
                try:
                    from predictors.neural.transformer_predictor import TransformerPredictor
                    d_model = clean_params.get("d_model", 32)
                    n_heads = clean_params.get("n_heads", 4)
                    if isinstance(d_model, float):
                        d_model = int(round(d_model))
                    if isinstance(n_heads, float):
                        n_heads = int(round(n_heads))
                    if d_model < 1:
                        d_model = 1
                    if n_heads < 1:
                        n_heads = 1
                    return TransformerPredictor(total, pick, d_model=d_model, n_heads=n_heads)
                except ImportError:
                    from predictors.frequency import FrequencyPredictor
                    return FrequencyPredictor(total, pick)

            elif name == "BayesianProbability":
                from predictors.bayesian_predictor import BayesianProbabilityPredictor
                return BayesianProbabilityPredictor(total, pick)

            elif name == "MonteCarlo":
                from predictors.montecarlo_predictor import MonteCarloPredictor
                n_sim = clean_params.get("n_simulations", 1000)
                if isinstance(n_sim, float):
                    n_sim = int(round(n_sim))
                if n_sim < 1:
                    n_sim = 1
                return MonteCarloPredictor(total, pick, n_simulations=n_sim)

            else:
                from predictors.frequency import FrequencyPredictor
                return FrequencyPredictor(total, pick)

        except Exception as e:
            from predictors.frequency import FrequencyPredictor
            return FrequencyPredictor(total, pick)

    def _save_results(self, results: Dict, field: int, initial_window: int):
        import numpy as np
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj if i is not None]
            return obj
        
        full_results = {
            "field": field,
            "method": "walk_forward",
            "initial_window": initial_window,
            "total_blocks": self.data.blocks_count(),
            "timestamp": datetime.now().isoformat(),
            "results": convert_to_serializable(results)
        }
        
        existing = []
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    existing = loaded if isinstance(loaded, list) else [loaded]
            except:
                existing = []
        
        existing.append(full_results)
        if len(existing) > 10:
            existing = existing[-10:]
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SAVE] Результаты сохранены в {self.results_file}")

    def _analyze_results(self, results: Dict, field: int):
        print("\n" + "=" * 80)
        print(f"📊 АНАЛИЗ РЕЗУЛЬТАТОВ WALK-FORWARD — ПОЛЕ {field}")
        print("=" * 80)
        
        if not results:
            print("Нет результатов для анализа")
            return
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]["best_score"], reverse=True)
        
        print("\n┌─────┬─────────────────────────────────────────┬─────────────┬──────────────────────────────┐")
        print("│ №   │ Модель                                  │ Точность    │ Лучшие параметры             │")
        print("├─────┼─────────────────────────────────────────┼─────────────┼──────────────────────────────┤")
        
        for i, (name, info) in enumerate(sorted_results, 1):
            score = info["best_score"]
            params = info["best_params"]
            param_str = str(params)[:28] if params else "{}"
            
            if score >= 0.3:
                score_str = f"⭐ {score:.3f}"
            elif score >= 0.15:
                score_str = f"👍 {score:.3f}"
            elif score > 0:
                score_str = f"   {score:.3f}"
            else:
                score_str = f"❌ 0.000"
            
            print(f"│ {i:2d}  │ {name:<39} │ {score_str:<11} │ {param_str:<28} │")
        
        print("└─────┴─────────────────────────────────────────┴─────────────┴──────────────────────────────┘")

    def _print_recommendations(self, results: Dict, field: int):
        print("\n" + "=" * 80)
        print(f"💡 РЕКОМЕНДАЦИИ — ПОЛЕ {field}")
        print("=" * 80)
        
        if not results:
            return
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]["best_score"], reverse=True)
        top5 = sorted_results[:5]
        
        print("\nТоп-5 моделей с лучшими параметрами:\n")
        for i, (name, info) in enumerate(top5, 1):
            score = info["best_score"]
            params = info["best_params"]
            print(f"{i}. {name}")
            print(f"   Точность: {score:.3f} ({score*100:.1f}%)")
            print(f"   Параметры: {params}")
            print()
        
        print("=" * 80)
        print("[INFO] Используйте пункт 15 для настройки параметров моделей")
        print("[INFO] Затем переобучите модели (пункт 11)")

    def show_iterations_log(self, model_name: str = None, field: int = 1):
        import os
        import json

        if not os.path.exists(self.results_file):
            print("[ERR] Нет сохранённых результатов")
            return

        with open(self.results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        latest = None
        if isinstance(all_results, list):
            for res in reversed(all_results):
                if res.get("field") == field:
                    latest = res
                    break
        else:
            latest = all_results

        if not latest:
            print(f"[ERR] Нет результатов для поля {field}")
            return

        results = latest.get("results", {})

        if model_name:
            if model_name not in results:
                print(f"[ERR] Модель {model_name} не найдена")
                return

            info = results[model_name]
            print(f"\n📋 ДЕТАЛЬНЫЙ ЛОГ: {model_name}")
            print("=" * 60)

            iterations = []
            if "iterations_file" in info and info["iterations_file"] and os.path.exists(info["iterations_file"]):
                try:
                    with open(info["iterations_file"], 'r', encoding='utf-8') as f:
                        iterations = json.load(f)
                    print(f"  [INFO] Загружено из файла: {info['iterations_file']}")
                except Exception as e:
                    print(f"  [WARN] Ошибка загрузки файла итераций: {e}")
                    iterations = info.get("iterations", [])
            else:
                iterations = info.get("iterations", [])

            for it in iterations:
                status = "✅" if it.get("correct") else "❌"
                pred = it.get("prediction")
                true_block = it.get("true_block")
                print(f"  Итер.{it['iteration']:3d}: {status} Прогноз={pred} → Реальность={true_block}")

            correct = sum(1 for it in iterations if it.get("correct"))
            total = len(iterations)
            if total > 0:
                print(f"\n  ИТОГО: {correct}/{total} ({correct/total*100:.1f}%)")
        else:
            print(f"\n📋 КРАТКИЙ ЛОГ ВСЕХ МОДЕЛЕЙ")
            print("=" * 60)

            for name, info in results.items():
                if "total_iterations" in info and "best_avg_matches" in info:
                    total = info.get("total_iterations", 0)
                    avg = info.get("best_avg_matches", 0)
                    correct = int(avg * total) if total > 0 else 0
                    print(f"  {name}: {correct}/{total} (сред.совп. {avg:.2f})")
                else:
                    iterations = info.get("iterations", [])
                    correct = sum(1 for it in iterations if it.get("correct"))
                    total = len(iterations)
                    if total > 0:
                        percentage = correct / total * 100
                        print(f"  {name}: {correct}/{total} ({percentage:.1f}%)")
                    else:
                        print(f"  {name}: нет данных")