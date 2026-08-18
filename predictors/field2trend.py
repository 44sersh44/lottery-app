import numpy as np
from typing import List
from .base import BasePredictor


class Field2TrendPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, lookback: int = 5, name: str = "Field2_Trend"):
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.lookback = lookback
        self.trend = 0.0
        self.last_value = None
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        values = [block[0] for block in blocks if block]
        if len(values) < 2:
            self.trend = 0.0
            self.last_value = values[-1] if values else 1
            self.is_trained = True
            return
        recent = values[-self.lookback:] if len(values) >= self.lookback else values
        x = np.arange(len(recent))
        y = np.array(recent)
        if len(recent) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        self.trend = slope
        self.last_value = values[-1]
        self.is_trained = True

    def _generate_combination_with_one_number(self, target_num: int) -> List[int]:
        result = [target_num]
        while len(result) < self.pick_count:
            new_num = np.random.randint(1, self.total_numbers + 1)
            if new_num not in result:
                result.append(new_num)
        return sorted(result)

    def predict_single(self) -> List[int]:
        if not self.is_trained:
            return self._fallback()
        predicted = self.last_value + self.trend * 1
        predicted = int(round(predicted))
        predicted = max(1, min(self.total_numbers, predicted))
        return self._generate_combination_with_one_number(predicted)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _fallback(self) -> List[int]:
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())