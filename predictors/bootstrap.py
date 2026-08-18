import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

class BootstrapPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, n_resamples: int = 100, name: str = "Bootstrap"):
        # ✅ ИСПРАВЛЕНО: передаём total_numbers и pick_count в базовый класс
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.n_resamples = n_resamples
        self.counter = None
        # Для двойной лотереи
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.blocks2 = []
        self.counter2 = None

    def fit(self, blocks: List[List[int]]) -> None:
        all_nums = [num for block in blocks for num in block]
        if not all_nums:
            self.counter = np.zeros(self.total_numbers)
        else:
            self.counter = np.zeros(self.total_numbers)
            for _ in range(self.n_resamples):
                sample = np.random.choice(all_nums, size=len(all_nums), replace=True)
                for num in sample:
                    self.counter[num-1] += 1
        self.is_trained = True

    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        """Обучение для двойной лотереи."""
        self.is_double = True
        self.blocks2 = blocks2
        self.pick_count2 = len(blocks2[0]) if blocks2 and blocks2[0] else 1
        self.total_numbers2 = max(max(block) for block in blocks2) if blocks2 else 54
        
        # Обучаем первое поле
        self.fit(blocks1)
        
        # Обучаем второе поле
        if blocks2:
            all_nums2 = [num for block in blocks2 for num in block]
            if all_nums2:
                self.counter2 = np.zeros(self.total_numbers2)
                for _ in range(self.n_resamples):
                    sample = np.random.choice(all_nums2, size=len(all_nums2), replace=True)
                    for num in sample:
                        self.counter2[num-1] += 1

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.is_double:
            return self._predict_double(n_predictions)
        return self._predict_single(n_predictions)

    def _predict_single(self, n_predictions: int) -> List[List[int]]:
        if self.counter is None or np.sum(self.counter) == 0:
            return self._fallback_single(n_predictions)
        
        top_indices = np.argsort(self.counter)[-self.pick_count:][::-1]
        base_pred = [i+1 for i in top_indices]
        predictions = []
        for offset in range(n_predictions):
            shifted = [(x + offset) % self.total_numbers + 1 for x in base_pred]
            predictions.append(sorted(shifted))
        return predictions

    def _predict_double(self, n_predictions: int) -> List[Tuple[List[int], List[int]]]:
        # Первое поле
        preds1 = self._predict_single(n_predictions)
        
        # Второе поле
        if self.counter2 is not None and np.sum(self.counter2) > 0:
            top2 = np.argsort(self.counter2)[-self.pick_count2:][::-1]
            base_pred2 = [i+1 for i in top2]
            preds2 = [[base_pred2[0]] for _ in range(n_predictions)]
        else:
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
        
        return [(preds1[i], preds2[i]) for i in range(n_predictions)]

    def predict_single(self) -> Union[List[int], Tuple[List[int], List[int]]]:
        result = self.predict(1)[0]
        return result

    def _fallback_single(self, n_predictions: int) -> List[List[int]]:
        return [sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
                for _ in range(n_predictions)]