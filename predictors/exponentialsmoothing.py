import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class ExponentialSmoothingPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, alpha: float = 0.3, name: str = "ExpSmoothing"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.alpha = alpha
        self.last_pred = None
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        sums = [sum(block) for block in blocks]
        if len(sums) == 0:
            self.last_pred = list(range(1, self.pick_count + 1))
            self.is_trained = True
            return
        exp_avg = sums[0]
        for s in sums[1:]:
            exp_avg = self.alpha * s + (1 - self.alpha) * exp_avg
        target_sum = int(round(exp_avg))
        self.last_pred = self._find_combo(target_sum)
        self.is_trained = True

    def _find_combo(self, target_sum: int) -> List[int]:
        for _ in range(1000):
            combo = sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) < 5:
                return combo
        return sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if not self.is_trained or self.last_pred is None:
            return self._fallback(n_predictions)

        if self.is_double:
            preds1 = [self.last_pred] * n_predictions
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
            return [(preds1[i], preds2[i]) for i in range(n_predictions)]

        return [self.last_pred] * n_predictions

    def predict_single(self) -> Union[List[int], Tuple[List[int], List[int]]]:
        result = self.predict(1)[0]
        return result

    def _fallback(self, n_predictions: int) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.is_double:
            preds1 = [sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())
                      for _ in range(n_predictions)]
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
            return [(preds1[i], preds2[i]) for i in range(n_predictions)]
        return [sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())
                for _ in range(n_predictions)]