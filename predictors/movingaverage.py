import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import Counter
from .base import BasePredictor

class MovingAveragePredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, window: int = 5, name: str = "MovingAverage"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.window = window
        self.sums = []
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.blocks2 = []

    def fit(self, blocks: List[List[int]]) -> None:
        self.sums = [sum(block) for block in blocks][-self.window:]
        self.is_trained = True

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.is_double:
            return self._predict_double(n_predictions)
        return self._predict_single(n_predictions)

    def _predict_single(self, n_predictions: int) -> List[List[int]]:
        if not self.sums:
            return self._fallback_single(n_predictions)
        target_sum = int(round(np.mean(self.sums)))
        predictions = []
        for _ in range(n_predictions):
            best = None
            best_diff = float('inf')
            for _ in range(1000):
                combo = sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
                diff = abs(sum(combo) - target_sum)
                if diff < best_diff:
                    best_diff = diff
                    best = combo
                if best_diff == 0:
                    break
            predictions.append(best)
        return predictions

    def _predict_double(self, n_predictions: int) -> List[Tuple[List[int], List[int]]]:
        preds1 = self._predict_single(n_predictions)
        preds2 = []
        if self.blocks2:
            counter = Counter()
            for block in self.blocks2:
                if block:
                    counter.update(block)
            if counter:
                most_common = counter.most_common(1)[0][0]
                preds2 = [[most_common] for _ in range(n_predictions)]
            else:
                preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
        else:
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
        return [(preds1[i], preds2[i]) for i in range(n_predictions)]

    def predict_single(self) -> List[int]:
        result = self.predict(1)[0]
        return result

    def _fallback_single(self, n_predictions: int) -> List[List[int]]:
        return [sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
                for _ in range(n_predictions)]