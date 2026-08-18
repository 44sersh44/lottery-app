import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class HarmonicMeanPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "HarmonicMean"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.harmonic_mean = None
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        all_nums = [num for block in blocks for num in block]
        if not all_nums:
            self.harmonic_mean = 1.0
        else:
            # Защита от деления на ноль
            inv_sum = sum(1.0 / x for x in all_nums if x != 0)
            if inv_sum == 0:
                self.harmonic_mean = 1.0
            else:
                self.harmonic_mean = len(all_nums) / inv_sum
        self.is_trained = True

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.harmonic_mean is None:
            return self._fallback(n_predictions)
        
        target = int(round(self.harmonic_mean))
        predictions = []
        for _ in range(n_predictions):
            combo = sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
            for i in range(len(combo)):
                if combo[i] > target:
                    combo[i] = max(combo[i]-1, 1)
                else:
                    combo[i] = min(combo[i]+1, self.total_numbers)
            predictions.append(sorted(combo))
        
        if self.is_double:
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
            return [(predictions[i], preds2[i]) for i in range(n_predictions)]
        
        return predictions

    def predict_single(self) -> Union[List[int], Tuple[List[int], List[int]]]:
        result = self.predict(1)[0]
        return result

    def _fallback(self, n_predictions: int) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.is_double:
            preds1 = [sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
                      for _ in range(n_predictions)]
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
            return [(preds1[i], preds2[i]) for i in range(n_predictions)]
        return [sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
                for _ in range(n_predictions)]