# services/prediction_service.py
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from core.predictor_engine import PredictorEngine
from core.parameter_manager import ParameterManager
from services.data_service import DataService
from analyzers.bayesian_montecarlo import BayesianAnalyzer, MonteCarloSimulator
from analyzers.prediction_filter import PredictionFilter
from analyzers.statistical_analyzer import StatisticalAnalyzer


class PredictionService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self._engine: Optional[PredictorEngine] = None

    @property
    def engine(self) -> PredictorEngine:
        if self._engine is None:
            data = self.data_service.current_data
            if data is None:
                raise ValueError("Нет активных данных для обучения")
            self._engine = PredictorEngine(data, self.data_service.current_profile_path)
        return self._engine

    def train_all(self, force_retrain: bool = False,
                  parallel: bool = False,
                  max_workers: Optional[int] = None,
                  model_names: Optional[List[str]] = None):
        self.engine.train_all(force_retrain=force_retrain, parallel=parallel,
                              max_workers=max_workers, model_names=model_names)

    def predict_all_models(self) -> Dict[str, Tuple[List[int], List[int]]]:
        return self.engine.predict_all_models()

    def evaluate_models(self, test_ratio: float = 0.2) -> Dict[str, float]:
        return self.engine.evaluate_models(test_ratio=test_ratio)

    def evaluate_models_detailed(self, test_ratio: float = 0.2) -> Dict[str, Dict]:
        return self.engine.evaluate_models_detailed(test_ratio=test_ratio)

    # ---------- Основные методы прогнозирования ----------

    def make_prediction(self, n: int, mode: str, ui) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        if data.blocks_count() < 3:
            raise ValueError("Недостаточно данных (нужно минимум 3 блока)")

        engine = self.engine
        engine.train_all(force_retrain=False)
        all_preds = engine.predict_all_models()
        detailed = engine.evaluate_models_detailed(test_ratio=0.2)
        is_double = data.is_double()

        if mode == '1':  # все модели
            weights = engine.evaluate_models(test_ratio=0.2)
            sorted_preds = sorted(all_preds.items(), key=lambda x: weights.get(x[0], 0), reverse=True)
            top = sorted_preds[:n]
            rows = []
            for i, (name, (p1, p2)) in enumerate(top, 1):
                avg = weights.get(name, 0)  # среднее количество угаданных чисел
                confidence_percent = (avg / data.pick_count) * 100 if data.pick_count > 0 else 0
                conf_str = f"{confidence_percent:.1f}%" if confidence_percent > 0 else "—"
                p1_str = ui.safe_format_block(p1)
                p2_str = ui.safe_format_block(p2) if is_double else "—"
                if is_double:
                    rows.append([str(i), name, p1_str, p2_str, conf_str])
                else:
                    rows.append([str(i), name, p1_str, conf_str])
            headers = ["№", "Модель", "Поле 1", "Поле 2", "Уверенность"] if is_double else ["№", "Модель", "Прогноз", "Уверенность"]
            return {'headers': headers, 'rows': rows, 'mode': 'all'}

        elif mode == '2':  # лучшие по рейтингу
            sorted_by_rating = sorted(
                [(name, stats) for name, stats in detailed.items() if stats.get('avg_sq', 0) > 0],
                key=lambda x: x[1]['avg_sq'], reverse=True
            )
            selected = sorted_by_rating[:n]
            rows = []
            for i, (name, stats) in enumerate(selected, 1):
                avg = stats.get('avg', 0)
                confidence_percent = (avg / data.pick_count) * 100 if data.pick_count > 0 else 0
                conf_str = f"{confidence_percent:.1f}%" if confidence_percent > 0 else "—"
                p1, p2 = all_preds.get(name, (None, None))
                p1_str = ui.safe_format_block(p1)
                p2_str = ui.safe_format_block(p2) if is_double else "—"
                if is_double:
                    rows.append([str(i), name, p1_str, p2_str, conf_str])
                else:
                    rows.append([str(i), name, p1_str, conf_str])
            headers = ["№", "Модель", "Поле 1", "Поле 2", "Уверенность"] if is_double else ["№", "Модель", "Прогноз", "Уверенность"]
            return {'headers': headers, 'rows': rows, 'mode': 'best'}

        else:  # режим 3: ансамбль из лучших
            sorted_by_rating = sorted(
                [(name, stats) for name, stats in detailed.items() if stats.get('avg_sq', 0) > 0],
                key=lambda x: x[1]['avg_sq'], reverse=True
            )
            top_n = min(10, len(sorted_by_rating))
            selected_names = [name for name, _ in sorted_by_rating[:top_n]]
            predictors = []
            weights = []
            for name in selected_names:
                if name in engine.predictors1:
                    predictors.append(engine.predictors1[name])
                    weights.append(detailed[name]['avg_sq'])
            if len(predictors) < 2:
                raise ValueError("Недостаточно моделей для ансамбля (нужно минимум 2)")

            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            valid_preds = []
            valid_weights = []
            for p, w in zip(predictors, weights):
                try:
                    pred = p.predict_single()
                    if data.is_double():
                        if isinstance(pred, tuple):
                            pred = pred[0]
                    if pred and len(pred) == data.pick_count:
                        valid_preds.append(pred)
                        valid_weights.append(w)
                except:
                    continue
            if len(valid_preds) < 2:
                raise ValueError("Недостаточно валидных прогнозов для ансамбля")

            weighted_votes = {}
            for i, pred in enumerate(valid_preds):
                w = valid_weights[i]
                for num in pred:
                    weighted_votes[num] = weighted_votes.get(num, 0) + w

            sorted_numbers = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
            results = []
            for offset in range(n):
                if offset + data.pick_count <= len(sorted_numbers):
                    candidate = [int(num) for num, _ in sorted_numbers[offset:offset + data.pick_count]]
                else:
                    candidate = [int(num) for num, _ in sorted_numbers[:data.pick_count-1]]
                    available = set(range(1, data.total_numbers+1)) - set(candidate)
                    if available:
                        candidate.append(random.choice(list(available)))
                candidate.sort()
                if candidate not in results:
                    results.append(candidate)
                if len(results) >= n:
                    break

            rows = [[str(i), ui.safe_format_block(pred)] for i, pred in enumerate(results, 1)]
            return {'headers': ["№", "Прогноз (ансамбль)"], 'rows': rows, 'mode': 'ensemble'}

    # ---------- POSITIONAL PREDICTION (с локальным импортом) ----------
    def positional_prediction(self) -> dict:
        from predictors.positional import PositionalPredictor   # <-- локальный импорт
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        total = data.total_numbers
        pick = data.pick_count
        blocks_for_train = list(reversed(data.blocks))
        pred = PositionalPredictor(total, pick)
        pred.fit(blocks_for_train)
        result1 = pred.predict_single()
        result2 = None
        if data.is_double():
            total2 = data.total_numbers2
            pick2 = data.pick_count2
            blocks2_for_train = list(reversed(data.blocks2)) if data.blocks2 else []
            if blocks2_for_train and len(blocks2_for_train) > 0:
                pred2 = PositionalPredictor(total2, pick2)
                pred2.fit(blocks2_for_train)
                result2 = pred2.predict_single()
            else:
                counter = Counter()
                for block in data.blocks2:
                    if block and len(block) > 0:
                        counter.update(block)
                if counter:
                    result2 = [counter.most_common(1)[0][0]]
                else:
                    result2 = [total2 // 2] if total2 else []
        return {
            'field1': result1,
            'field2': result2,
            'is_double': data.is_double()
        }

    # ---------- COMPARE PREDICTION ----------
    def compare_prediction(self, real_numbers: List[int], real_numbers2: Optional[List[int]] = None) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.engine
        engine.train_all(force_retrain=False)
        all_preds = engine.predict_all_models()
        detailed = engine.evaluate_models_detailed(test_ratio=0.2)
        is_double = data.is_double()

        results = []
        for model_name, (p1, p2) in all_preds.items():
            if not p1 or len(p1) != data.pick_count:
                continue
            matches1 = len(set(p1) & set(real_numbers))
            if is_double and p2 and len(p2) == data.pick_count2:
                matches2 = len(set(p2) & set(real_numbers2)) if real_numbers2 else 0
            else:
                matches2 = 0
            total_matches = matches1 + matches2
            stats = detailed.get(model_name, {})
            avg = stats.get('avg', None)
            avg_sq = stats.get('avg_sq', None)
            results.append({
                'model': model_name,
                'p1': p1,
                'p2': p2 if p2 else [],
                'matches1': matches1,
                'matches2': matches2,
                'total': total_matches,
                'avg': avg,
                'avg_sq': avg_sq
            })

        results.sort(key=lambda x: x['total'], reverse=True)
        rows = []
        for i, item in enumerate(results[:20], 1):
            p1_str = ' '.join(str(x) for x in item['p1'])
            if is_double:
                p2_str = ' '.join(str(x) for x in item['p2']) if item['p2'] else "—"
                avg_str = f"{item['avg']:.2f}" if item['avg'] is not None else "—"
                avg_sq_str = f"{item['avg_sq']:.2f}" if item['avg_sq'] is not None else "—"
                rows.append([str(i), item['model'], p1_str, p2_str,
                             str(item['matches1']), str(item['matches2']),
                             avg_str, avg_sq_str])
            else:
                avg_str = f"{item['avg']:.2f}" if item['avg'] is not None else "—"
                avg_sq_str = f"{item['avg_sq']:.2f}" if item['avg_sq'] is not None else "—"
                rows.append([str(i), item['model'], p1_str,
                             str(item['matches1']), avg_str, avg_sq_str])

        if is_double:
            headers = ["№", "Модель", "Поле 1", "Поле 2", "Совп.1", "Совп.2", "Среднее", "Рейтинг"]
        else:
            headers = ["№", "Модель", "Прогноз", "Совпадений", "Среднее", "Рейтинг"]

        best = results[0] if results else None
        return {
            'headers': headers,
            'rows': rows,
            'best_model': best['model'] if best else None,
            'is_double': is_double
        }

    # ---------- SHOW ALL MODEL PREDICTIONS ----------
    def show_all_model_predictions(self) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.engine
        engine.train_all(force_retrain=False)
        all_preds = engine.predict_all_models()
        detailed = engine.evaluate_models_detailed(test_ratio=0.2)
        is_double = data.is_double()

        items = []
        for name, (p1, p2) in all_preds.items():
            stats = detailed.get(name, {})
            avg_sq = stats.get('avg_sq', -1.0)
            avg = stats.get('avg', -1.0)
            max_m = stats.get('max', 0)
            dist = stats.get('dist', {})
            dist_str = ', '.join(f"{k}:{v}" for k, v in sorted(dist.items()))
            items.append({
                'name': name,
                'p1': p1,
                'p2': p2 if p2 else [],
                'avg_sq': avg_sq,
                'avg': avg,
                'max': max_m,
                'dist': dist_str
            })
        items.sort(key=lambda x: x['avg_sq'], reverse=True)

        rows = []
        for i, item in enumerate(items, 1):
            p1_str = ' '.join(str(x) for x in item['p1'])
            p2_str = ' '.join(str(x) for x in item['p2']) if item['p2'] else "—"
            avg_sq_str = f"{item['avg_sq']:.2f}" if item['avg_sq'] >= 0 else "—"
            max_str = str(item['max']) if item['max'] > 0 else "—"
            if is_double:
                rows.append([str(i), item['name'], p1_str, p2_str, avg_sq_str, max_str])
            else:
                rows.append([str(i), item['name'], p1_str, avg_sq_str, max_str])

        if is_double:
            headers = ["№", "Модель", "Поле 1", "Поле 2", "Рейтинг (ср.кв.)", "Макс."]
        else:
            headers = ["№", "Модель", "Прогноз", "Рейтинг (ср.кв.)", "Макс."]

        best = items[0] if items else None
        return {
            'headers': headers,
            'rows': rows,
            'best_name': best['name'] if best else None,
            'best_dist': best['dist'] if best else None,
            'is_double': is_double
        }

    # ---------- FILTERED PREDICTION ----------
    def filtered_prediction(self, n: int) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.engine
        engine.train_all(force_retrain=False)
        predictions = engine.predict_top_n(n * 3)
        analyzer = StatisticalAnalyzer(data.blocks, data.total_numbers, data.pick_count)
        filter_tool = PredictionFilter(analyzer)
        filtered = filter_tool.get_best_predictions(predictions, n)

        rows = []
        for i, (p1, p2, conf) in enumerate(filtered, 1):
            p1_str = ' '.join(str(x) for x in p1)
            if data.is_double():
                p2_str = ' '.join(str(x) for x in p2) if p2 else "—"
                rows.append([str(i), p1_str, p2_str, f"{conf:.1f}%"])
            else:
                rows.append([str(i), p1_str, f"{conf:.1f}%"])

        if data.is_double():
            headers = ["№", "Поле 1", "Поле 2", "Уверенность"]
        else:
            headers = ["№", "Прогноз", "Уверенность"]
        return {'headers': headers, 'rows': rows, 'is_double': data.is_double()}

    # ---------- BAYESIAN ANALYSIS ----------
    def bayesian_analysis(self, field: int) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        if field == 1:
            blocks = data.blocks
            total = data.total_numbers
            pick = data.pick_count
        else:
            blocks = data.blocks2
            total = data.total_numbers2
            pick = data.pick_count2
        analyzer = BayesianAnalyzer(blocks, total, pick)
        report = analyzer.get_report()
        predictions = analyzer.predict_combination(5)
        return {
            'report': report,
            'predictions': predictions,
            'field': field
        }

    # ---------- MONTE CARLO SIMULATION ----------
    def monte_carlo_simulation(self, field: int, sim_type: str, n: int, prediction: Optional[List[int]] = None) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        if field == 1:
            blocks = data.blocks
            total = data.total_numbers
            pick = data.pick_count
        else:
            blocks = data.blocks2
            total = data.total_numbers2
            pick = data.pick_count2
        simulator = MonteCarloSimulator(blocks, total, pick)

        if sim_type == 'historical':
            combos = simulator.find_best_combinations(5, n, method='historical')
            return {'type': 'historical', 'combos': combos}
        elif sim_type == 'markov':
            combos = simulator.find_best_combinations(5, n, method='markov')
            return {'type': 'markov', 'combos': combos}
        elif sim_type == 'probability' and prediction:
            probs = simulator.estimate_win_probability(prediction, n)
            return {'type': 'probability', 'probs': probs}
        elif sim_type == 'best':
            combos = simulator.find_best_combinations(n, n*100, method='historical')
            return {'type': 'best', 'combos': combos}
        else:
            raise ValueError("Неизвестный тип симуляции")

    # ---------- CREATE ENSEMBLE FROM BEST (с локальным импортом) ----------
    def create_ensemble_from_best(self, top_n: int = None, threshold: float = None,
                                  manual_names: List[str] = None,
                                  ensemble_name: str = None) -> str:
        """
        Создаёт ансамбль из лучших моделей по рейтингу.
        Если ensemble_name не указан, генерируется автоматически с временной меткой.
        """
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        
        engine = self.engine
        engine.train_all(force_retrain=False)
        detailed = engine.evaluate_models_detailed(test_ratio=0.2)
        
        # Сортируем модели по рейтингу (avg_sq)
        sorted_models = sorted(
            [(name, stats) for name, stats in detailed.items() if stats.get('avg_sq', 0) > 0],
            key=lambda x: x[1]['avg_sq'], reverse=True
        )
        
        # Выбираем модели по заданному критерию
        if manual_names:
            selected_names = [n for n in manual_names if n in [m[0] for m in sorted_models]]
        elif threshold is not None:
            selected_names = [name for name, stats in sorted_models if stats['avg_sq'] >= threshold]
        else:
            top_n = top_n or 10
            selected_names = [name for name, _ in sorted_models[:top_n]]
        
        if len(selected_names) < 2:
            raise ValueError("Недостаточно моделей для ансамбля (нужно минимум 2)")
        
        # Собираем предикторы в словарь {имя: модель}
        predictors_dict = {}
        for name in selected_names:
            if name in engine.predictors1:
                predictors_dict[name] = engine.predictors1[name]
        
        if len(predictors_dict) < 2:
            raise ValueError("Недостаточно предикторов")
        
        # Импортируем EnsemblePredictor
        from predictors.ensemble import EnsemblePredictor
        
        # Веса = рейтинг (avg_sq)
        weights = {name: detailed[name]['avg_sq'] for name in predictors_dict.keys()}
        
        # Если имя не указано – генерируем
        if ensemble_name is None:
            from datetime import datetime
            ensemble_name = f"Ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Создаём ансамбль
        ensemble = EnsemblePredictor(predictors=predictors_dict, weights=weights, name=ensemble_name)
        
        # Устанавливаем атрибуты, если не установлены
        if not hasattr(ensemble, 'total_numbers') or ensemble.total_numbers is None:
            ensemble.total_numbers = data.total_numbers
        if not hasattr(ensemble, 'pick_count') or ensemble.pick_count is None:
            ensemble.pick_count = data.pick_count
        
        # Обучаем ансамбль на данных
        blocks_for_train = list(reversed(data.blocks))
        ensemble.fit(blocks_for_train)
        
        # Сохраняем в движок и в кэш (используем динамическое имя)
        engine.predictors1[ensemble_name] = ensemble
        if engine.model_manager:
            engine.model_manager.save_model(ensemble_name, ensemble, field=1)
        
        return ensemble_name

    # ---------- COMPARE FORECASTS ----------
    def compare_forecasts(self) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.engine
        engine.train_all(force_retrain=False)
        all_preds = engine.predict_all_models()
        is_double = data.is_double()

        groups = {}
        for model_name, (p1, p2) in all_preds.items():
            if not p1:
                continue
            key = (tuple(sorted(p1)), tuple(sorted(p2)) if is_double and p2 else ())
            if key not in groups:
                groups[key] = {'count': 0, 'models': []}
            groups[key]['count'] += 1
            groups[key]['models'].append(model_name)

        items = []
        for (p1, p2), info in groups.items():
            items.append({
                'p1': list(p1),
                'p2': list(p2) if is_double else None,
                'count': info['count'],
                'models': info['models']
            })
        items.sort(key=lambda x: x['count'], reverse=True)

        rows = []
        for i, item in enumerate(items, 1):
            p1_str = ' '.join(str(x) for x in item['p1'])
            p2_str = ' '.join(str(x) for x in item['p2']) if is_double else '—'
            models_str = ', '.join(item['models'][:5])
            if len(item['models']) > 5:
                models_str += f" и ещё {len(item['models']) - 5}"
            rows.append([str(i), p1_str, p2_str, str(item['count']), models_str])

        headers = ["№", "Поле 1", "Поле 2", "Кол-во моделей", "Модели"]
        return {'headers': headers, 'rows': rows, 'is_double': is_double}

    # ---------- NUMBER CONSENSUS ----------
    def get_number_consensus(self, top_n: int = 15) -> dict:
        data = self.data_service.current_data
        if not data:
            raise ValueError("Нет данных")
        engine = self.engine
        engine.train_all(force_retrain=False)
        all_preds = engine.predict_all_models()

        total_models = len(all_preds)
        models_by_number = {}
        for model_name, (p1, p2) in all_preds.items():
            if p1:
                for num in p1:
                    if num not in models_by_number:
                        models_by_number[num] = []
                    models_by_number[num].append(model_name)

        items = []
        for num, models_list in models_by_number.items():
            count = len(models_list)
            percent = (count / total_models) * 100 if total_models > 0 else 0
            items.append({
                'num': num,
                'count': count,
                'percent': percent,
                'models': models_list
            })
        items.sort(key=lambda x: x['count'], reverse=True)

        rows = []
        for i, item in enumerate(items[:top_n], 1):
            models_str = ', '.join(item['models'])
            rows.append([str(i), str(item['num']), str(item['count']), f"{item['percent']:.1f}%", models_str])

        headers = ["№", "Число", "Кол-во моделей", "% моделей", "Модели"]
        return {'headers': headers, 'rows': rows}