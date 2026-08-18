import numpy as np
from typing import List
from .base import BasePredictor


class Field2LinearPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "Field2_Linear"):
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.coef = 0.0
        self.intercept = 0.0
        self.last_idx = 0
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        values = [block[0] for block in blocks if block]
        if len(values) < 2:
            self.coef = 0.0
            self.intercept = values[0] if values else 1
            self.last_idx = len(values)
            self.is_trained = True
            return
        x = np.arange(len(values))
        y = np.array(values)
        self.coef, self.intercept = np.polyfit(x, y, 1)
        self.last_idx = len(values)
        self.is_trained = True

    def _generate_combination_with_one_number(self, target_num: int) -> List[int]:
        # Генерируем комбинацию, в которой одно число равно target_num
        result = [target_num]
        while len(result) < self.pick_count:
            new_num = np.random.randint(1, self.total_numbers + 1)
            if new_num not in result:
                result.append(new_num)
        return sorted(result)

    def predict_single(self) -> List[int]:
        if not self.is_trained:
            return self._fallback()
        predicted = self.coef * (self.last_idx + 1) + self.intercept
        predicted = int(round(predicted))
        predicted = max(1, min(self.total_numbers, predicted))
        return self._generate_combination_with_one_number(predicted)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _fallback(self) -> List[int]:
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())