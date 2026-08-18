"""Ансамбль из нескольких лучших предикторов (голосование)."""

from typing import List, Dict, Union, Tuple
from collections import defaultdict


class EnsemblePredictor:
    def __init__(self, predictors: Dict[str, 'BasePredictor'], weights: Dict[str, float] = None, name: str = "Ensemble"):
        self.predictors = predictors
        self.weights = weights if weights else {name: 1.0 for name in predictors}
        self.name = name
        self.is_trained = True
        self.is_double = False
        self.total_numbers = None
        self.total_numbers2 = None
        self.pick_count = None
        self.pick_count2 = None

        # Определяем параметры из первой модели
        for pred in self.predictors.values():
            if hasattr(pred, 'total_numbers'):
                self.total_numbers = pred.total_numbers
            if hasattr(pred, 'pick_count'):
                self.pick_count = pred.pick_count
            if hasattr(pred, 'is_double') and pred.is_double:
                self.is_double = True
                self.total_numbers2 = getattr(pred, 'total_numbers2', 54)
                self.pick_count2 = getattr(pred, 'pick_count2', 1)
                break

        if self.pick_count is None:
            self.pick_count = 6
        if self.total_numbers is None:
            self.total_numbers = 45
        if self.pick_count2 is None:
            self.pick_count2 = 1
        if self.total_numbers2 is None:
            self.total_numbers2 = 54

    def fit(self, blocks: List[List[int]]) -> None:
        for pred in self.predictors.values():
            if not pred.is_trained:
                pred.fit(blocks)
        self.is_trained = True

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")

        if self.is_double:
            return self._predict_double(n_predictions)
        else:
            return self._predict_single(n_predictions)

    def _predict_single(self, n_predictions: int) -> List[List[int]]:
        from collections import defaultdict
        score_counter = defaultdict(float)

        for name, pred in self.predictors.items():
            try:
                preds = pred.predict(n_predictions)
                weight = self.weights.get(name, 1.0)
                for i, p in enumerate(preds):
                    if isinstance(p, tuple):
                        p = p[0]
                    if p and isinstance(p, list):
                        for num in p:
                            score_counter[num] += weight
            except Exception:
                continue

        if not score_counter:
            import numpy as np
            return [sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())
                    for _ in range(n_predictions)]

        selected = [num for num, _ in sorted(score_counter.items(), key=lambda x: x[1], reverse=True)[:self.pick_count]]
        selected.sort()
        return [selected] * n_predictions

    def _predict_double(self, n_predictions: int) -> List[Tuple[List[int], List[int]]]:
        from collections import defaultdict
        score_counter1 = defaultdict(float)
        score_counter2 = defaultdict(float)

        for name, pred in self.predictors.items():
            try:
                preds = pred.predict(n_predictions)
                weight = self.weights.get(name, 1.0)
                for p in preds:
                    if isinstance(p, tuple) and len(p) == 2:
                        p1, p2 = p
                        if p1 and isinstance(p1, list):
                            for num in p1:
                                score_counter1[num] += weight
                        if p2 and isinstance(p2, list):
                            for num in p2:
                                score_counter2[num] += weight
                    elif isinstance(p, list):
                        for num in p:
                            score_counter1[num] += weight
            except Exception:
                continue

        if score_counter1:
            result1 = [num for num, _ in sorted(score_counter1.items(), key=lambda x: x[1], reverse=True)[:self.pick_count]]
            result1.sort()
        else:
            result1 = self._fallback_field(self.total_numbers, self.pick_count)

        if score_counter2:
            result2 = [num for num, _ in sorted(score_counter2.items(), key=lambda x: x[1], reverse=True)[:self.pick_count2]]
            result2.sort()
        else:
            result2 = self._fallback_second_field()

        if len(result1) < self.pick_count:
            result1 = self._fill_missing(result1, self.total_numbers, self.pick_count)

        return [(result1, result2)] * n_predictions

    def _fill_missing(self, current: List[int], total_numbers: int, pick_count: int) -> List[int]:
        all_nums = set(range(1, total_numbers + 1))
        used = set(current)
        remaining = list(all_nums - used)
        if remaining:
            import random
            random.shuffle(remaining)
            current.extend(remaining[:pick_count - len(current)])
            current.sort()
        return current

    def _fallback_field(self, total_numbers: int, pick_count: int) -> List[int]:
        import numpy as np
        return sorted(np.random.choice(range(1, total_numbers + 1), pick_count, replace=False).tolist())

    def _fallback_second_field(self) -> List[int]:
        return [self.total_numbers2 // 2]

    def predict_single(self) -> Union[List[int], Tuple[List[int], List[int]]]:
        result = self.predict(1)[0]
        return result