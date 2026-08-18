from .base import BasePredictor
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class BayesianPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, alpha: float = 0.5, **kwargs):
        # ПРАВИЛЬНЫЙ ВЫЗОВ super
        super().__init__(total_numbers, pick_count)
        self.alpha = alpha
        self.probs = None
        # Для двойной лотереи
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.blocks2 = []
        self.probs2 = None
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        counter = np.zeros(self.total_numbers)
        for block in blocks:
            for num in block:
                counter[num-1] += 1
        total = len(blocks) * self.pick_count
        self.probs = (counter + self.alpha) / (total + self.alpha * self.total_numbers)
        self.is_trained = True
    
    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        """Обучение для двойной лотереи."""
        self.is_double = True
        self.blocks2 = blocks2
        
        # Обучаем первое поле
        self.fit(blocks1)
        
        # Обучаем второе поле
        if blocks2:
            counter2 = np.zeros(self.total_numbers2)
            for block in blocks2:
                for num in block:
                    counter2[num-1] += 1
            total2 = len(blocks2) * self.pick_count2
            self.probs2 = (counter2 + self.alpha) / (total2 + self.alpha * self.total_numbers2)

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if not self.is_trained:
            return self._fallback_single(n_predictions)
        if self.is_double:
            return self._predict_double(n_predictions)
        return self._predict_single(n_predictions)

    def _predict_single(self, n_predictions: int) -> List[List[int]]:
        if self.probs is None or np.isnan(self.probs).any():
            return self._fallback_single(n_predictions)
        
        top_indices = np.argsort(self.probs)[-self.pick_count:][::-1]
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
        if self.probs2 is not None:
            top2 = np.argsort(self.probs2)[-self.pick_count2:][::-1]
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