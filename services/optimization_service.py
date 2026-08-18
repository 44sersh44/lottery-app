# services/optimization_service.py
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime
from collections import Counter

from core.hyperparameter_tuner import HyperparameterTuner
from core.parameter_manager import ParameterManager
from core.predictor_engine import PredictorEngine
from services.data_service import DataService
from services.prediction_service import PredictionService

class OptimizationService:
    def __init__(self, data_service: DataService, prediction_service: PredictionService):
        self.data_service = data_service
        self.prediction_service = prediction_service

    def backtest(self, test_ratio: float = 0.2) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.prediction_service.engine
        engine.train_all(force_retrain=True)
        results1 = engine.evaluate_models(test_ratio=test_ratio)
        results2 = {}
        if data.is_double():
            results2 = engine.evaluate_model_on_field2(test_ratio=test_ratio)
        return {'field1': results1, 'field2': results2}

    def walk_forward_tuning(self, field: int, step: str, adaptive: bool,
                            initial_window: Optional[int] = None,
                            use_current_params: bool = True) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        if data.blocks_count() < 10:
            raise ValueError("Недостаточно данных (минимум 10 блоков)")
        if initial_window is None:
            initial_window = int(data.blocks_count() * 0.8)
        if initial_window >= data.blocks_count():
            raise ValueError("Начальное окно не может быть >= общего числа блоков")
        tuner = HyperparameterTuner(data, self.data_service.current_profile_path)
        results = tuner.run_tuning(
            field,
            initial_window / data.blocks_count(),
            step,
            use_current_params=use_current_params,
            adaptive=adaptive
        )
        return results

    def show_walk_forward_log(self, field: int, model_name: Optional[str] = None):
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        tuner = HyperparameterTuner(data, self.data_service.current_profile_path)
        tuner.show_iterations_log(model_name, field)

    def apply_best_models_by_threshold(self, field: int, target_digits: int) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        results_file = os.path.join(self.data_service.current_profile_path, "walk_forward_results.json")
        if not os.path.exists(results_file):
            raise FileNotFoundError("Сначала запустите Walk-Forward оптимизацию (пункт 17)")

        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        latest = None
        if isinstance(all_results, list):
            for res in reversed(all_results):
                if res.get("field") == field:
                    latest = res
                    break
        else:
            latest = all_results if all_results.get("field") == field else None
        if not latest:
            raise ValueError(f"Нет результатов для поля {field}")

        results = latest.get("results", {})
        stats = {d: [] for d in range(1, 8)}
        for name, info in results.items():
            matches_list = info.get("matches_list", [])
            if matches_list:
                dist = Counter(matches_list)
                for d in range(1, 8):
                    if dist.get(d, 0) > 0:
                        stats[d].append((name, dist[d], info))

        selected = stats.get(target_digits, [])
        if not selected:
            raise ValueError(f"Нет моделей, угадавших ровно {target_digits} цифр")

        pm = ParameterManager(self.data_service.current_profile_path)
        applied = 0
        for name, count, info in selected:
            best_params = info.get("best_params", {})
            if not best_params:
                continue
            full_name = name if field == 1 else f"{name}_field2"
            for pname, pval in best_params.items():
                pm.set_param(full_name, pname, pval)
            applied += 1

        return {
            'applied': applied,
            'selected_models': [name for name, _, _ in selected],
            'target_digits': target_digits,
            'field': field
        }

    def walk_forward_forecast(self, initial_window: Optional[int] = None,
                              n_forecasts: int = 1,
                              criterion: str = 'rating',
                              use_parallel: bool = False,
                              max_workers: Optional[int] = None) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        total_blocks = data.blocks_count()
        if total_blocks < 10:
            raise ValueError("Недостаточно данных (минимум 10 блоков)")

        if initial_window is None:
            initial_window = int(total_blocks * 0.8)
        if initial_window >= total_blocks:
            raise ValueError("Начальное окно не может быть >= общего числа блоков")

        is_double = data.is_double()
        # прогон по всем окнам
        model_stats = {}
        for train_size in range(initial_window, total_blocks):
            train_blocks1 = data.blocks[:train_size]
            test_block1 = data.blocks[train_size]
            train_blocks2 = data.blocks2[:train_size] if is_double else []
            test_block2 = data.blocks2[train_size] if is_double else None

            # создаём временный объект данных
            from core.lottery_data import LotteryData
            temp_data = LotteryData()
            temp_data.lottery_type = data.lottery_type
            temp_data.total_numbers = data.total_numbers
            temp_data.pick_count = data.pick_count
            temp_data.total_numbers2 = data.total_numbers2
            temp_data.pick_count2 = data.pick_count2
            temp_data.blocks = train_blocks1
            if is_double:
                temp_data.blocks2 = train_blocks2
            # копируем метаданные, если нужно
            temp_data.metadata = data.metadata[:train_size]

            engine = PredictorEngine(temp_data, self.data_service.current_profile_path)
            engine.train_all(force_retrain=True, parallel=use_parallel, max_workers=max_workers)
            all_preds = engine.predict_all_models()

            for model_name, (p1, p2) in all_preds.items():
                matches1 = len(set(p1) & set(test_block1)) if p1 else 0
                matches2 = 0
                if is_double and p2 and test_block2:
                    if len(p2) == 1:
                        matches2 = 1 if p2[0] == test_block2[0] else 0
                    else:
                        matches2 = len(set(p2) & set(test_block2))
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'field1': {'total_matches': 0, 'total_tests': 0, 'dist': {}},
                        'field2': {'total_matches': 0, 'total_tests': 0, 'dist': {}}
                    }
                model_stats[model_name]['field1']['total_matches'] += matches1
                model_stats[model_name]['field1']['total_tests'] += 1
                model_stats[model_name]['field1']['dist'][matches1] = \
                    model_stats[model_name]['field1']['dist'].get(matches1, 0) + 1
                if is_double:
                    model_stats[model_name]['field2']['total_matches'] += matches2
                    model_stats[model_name]['field2']['total_tests'] += 1
                    model_stats[model_name]['field2']['dist'][matches2] = \
                        model_stats[model_name]['field2']['dist'].get(matches2, 0) + 1

        # формируем таблицу статистики
        table_data = []
        for model_name, stats in model_stats.items():
            avg1 = stats['field1']['total_matches'] / stats['field1']['total_tests'] if stats['field1']['total_tests'] > 0 else 0
            dist1 = stats['field1']['dist']
            total_sq1 = sum(k * k * cnt for k, cnt in dist1.items())
            avg_sq1 = total_sq1 / stats['field1']['total_tests'] if stats['field1']['total_tests'] > 0 else 0
            dist_str1 = ', '.join(f"{k}:{dist1.get(k,0)}" for k in sorted(dist1.keys()))
            if is_double:
                avg2 = stats['field2']['total_matches'] / stats['field2']['total_tests'] if stats['field2']['total_tests'] > 0 else 0
                dist2 = stats['field2']['dist']
                total_sq2 = sum(k * k * cnt for k, cnt in dist2.items())
                avg_sq2 = total_sq2 / stats['field2']['total_tests'] if stats['field2']['total_tests'] > 0 else 0
                dist_str2 = ', '.join(f"{k}:{dist2.get(k,0)}" for k in sorted(dist2.keys()))
            else:
                avg2 = None
                avg_sq2 = None
                dist_str2 = None
            table_data.append({
                'model': model_name,
                'avg1': avg1,
                'avg_sq1': avg_sq1,
                'dist1': dist_str1,
                'avg2': avg2,
                'avg_sq2': avg_sq2,
                'dist2': dist_str2,
                'tests1': stats['field1']['total_tests'],
                'tests2': stats['field2']['total_tests'] if is_double else 0
            })

        # сортировка по рейтингу
        table_data.sort(key=lambda x: x['avg_sq1'], reverse=True)

        # финальное обучение на всех данных
        final_engine = PredictorEngine(data, self.data_service.current_profile_path)
        final_engine.train_all(force_retrain=True, parallel=use_parallel, max_workers=max_workers)
        final_preds = final_engine.predict_all_models()

        # отбор моделей по критерию
        if criterion == 'avg':
            sorted_by_avg = sorted(table_data, key=lambda x: x['avg1'], reverse=True)
            selected = sorted_by_avg[:n_forecasts]
        elif criterion == 'both':
            # берём из рейтинга и из среднего
            combined = []
            for item in table_data[:n_forecasts]:
                if item not in combined:
                    combined.append(item)
            for item in sorted(table_data, key=lambda x: x['avg1'], reverse=True)[:n_forecasts]:
                if item not in combined:
                    combined.append(item)
            selected = combined[:n_forecasts*2]
        else:  # rating
            selected = table_data[:n_forecasts]

        forecasts = []
        for item in selected:
            model_name = item['model']
            if model_name in final_preds:
                p1, p2 = final_preds[model_name]
                forecasts.append({
                    'model': model_name,
                    'p1': p1,
                    'p2': p2,
                    'avg1': item['avg1'],
                    'avg_sq1': item['avg_sq1'],
                    'dist1': item['dist1']
                })

        return {
            'stats_table': table_data,
            'forecasts': forecasts,
            'is_double': is_double,
            'criterion': criterion
        }

    def run_local_tuning(self, fields_processed: List[int], auto_save: bool = False) -> dict:
        """Локальная оптимизация параметров после Walk-Forward."""
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        from core.hyperparameter_tuner import HyperparameterTuner
        from core.parameter_manager import ParameterManager
        import itertools

        pm = ParameterManager(self.data_service.current_profile_path)
        tuner = HyperparameterTuner(data, self.data_service.current_profile_path)
        base_grid = tuner.PARAM_GRIDS

        # определяем модели для каждого поля
        all_models = set()
        for field in fields_processed:
            if field == 1:
                models = [name for name in pm.params.keys() if not name.endswith("_field2")]
            else:
                models = [name for name in pm.params.keys() if name.endswith("_field2")]
            all_models.update(models)

        improved_count = 0
        total_checked = 0
        details = {}

        for model_name in all_models:
            base_name = model_name.replace("_field2", "")
            if base_name not in base_grid or not base_grid[base_name]:
                continue
            param_grid = base_grid[base_name]
            # текущие параметры
            current_params = {}
            for pname in param_grid:
                val = pm.get_param(model_name, pname, None)
                if val is not None:
                    current_params[pname] = val
                else:
                    current_params[pname] = param_grid[pname][0] if param_grid[pname] else None
            if not current_params or any(v is None for v in current_params.values()):
                continue

            # шаги для параметров
            step_map = {
                "order": 1, "decay": 0.01, "k": 1, "window": 1, "alpha": 0.1,
                "n_resamples": 10, "n_estimators": 5, "learning_rate": 0.01,
                "max_depth": 1, "min_weight": 0.001, "C": 0.1,
                "eps": 0.05, "min_samples": 1, "contamination": 0.01,
                "n_nonzero_coefs": 1, "population_size": 5, "generations": 5,
                "mutation_rate": 0.01, "lookback": 1, "epochs": 5,
                "d_model": 4, "n_heads": 1, "num_layers": 1, "cv_folds": 1,
            }
            categorical = {"gamma", "seasonality_mode", "use_features", "meta_model",
                           "model_selection", "strategy", "voting", "performance_based", "use_ranking"}

            param_values = {}
            for pname, current_val in current_params.items():
                if pname in categorical:
                    param_values[pname] = param_grid.get(pname, [current_val])
                elif pname in step_map:
                    step = step_map[pname]
                    if isinstance(current_val, (int, float)):
                        values = []
                        for i in range(-5, 6):
                            new_val = current_val + i * step
                            if pname in param_grid and param_grid[pname]:
                                grid_vals = param_grid[pname]
                                if isinstance(grid_vals, list) and all(isinstance(x, (int, float)) for x in grid_vals):
                                    min_val = min(grid_vals)
                                    max_val = max(grid_vals)
                                    if min_val <= new_val <= max_val:
                                        values.append(new_val)
                                else:
                                    values = [current_val]
                                    break
                            else:
                                values.append(new_val)
                        values = sorted(set(values + [current_val]))
                        param_values[pname] = values
                    else:
                        param_values[pname] = [current_val]
                else:
                    param_values[pname] = [current_val]

            keys = list(param_values.keys())
            values = [param_values[k] for k in keys]
            combinations = list(itertools.product(*values))
            if len(combinations) > 50:
                step_idx = len(combinations) // 50
                combinations = combinations[::max(1, step_idx)][:50]

            field = 1 if not model_name.endswith("_field2") else 2
            current_avg = tuner.evaluate_combination(base_name, current_params, field=field)

            best_avg = -1.0
            best_combo = None
            for combo in combinations:
                params = dict(zip(keys, combo))
                avg = tuner.evaluate_combination(base_name, params, field=field)
                if avg > best_avg:
                    best_avg = avg
                    best_combo = params

            if best_combo and best_avg > current_avg:
                improved_count += 1
                details[model_name] = {'old': current_params, 'new': best_combo, 'old_avg': current_avg, 'new_avg': best_avg}
                if auto_save:
                    for pname, pval in best_combo.items():
                        pm.set_param(model_name, pname, pval)
            total_checked += 1

        return {
            'total_checked': total_checked,
            'improved_count': improved_count,
            'details': details,
            'auto_saved': auto_save
        }