import random
from collections import defaultdict, Counter
from typing import List
from .base import BasePredictor


class Field2MarkovPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "Field2_Markov"):
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.transitions = defaultdict(Counter)
        self.last_value = None
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.transitions.clear()
        values = [block[0] for block in blocks if block]
        for i in range(len(values) - 1):
            current = values[i]
            next_val = values[i + 1]
            self.transitions[current][next_val] += 1
        self.last_value = values[-1] if values else 1
        self.is_trained = True

    def _generate_combination_with_one_number(self, target_num: int) -> List[int]:
        result = [target_num]
        while len(result) < self.pick_count:
            new_num = random.randint(1, self.total_numbers)
            if new_num not in result:
                result.append(new_num)
        return sorted(result)

    def predict_single(self) -> List[int]:
        if not self.is_trained:
            return self._fallback()
        current = self.last_value
        if current in self.transitions:
            choices = list(self.transitions[current].keys())
            weights = list(self.transitions[current].values())
            next_val = random.choices(choices, weights=weights)[0]
        else:
            next_val = random.randint(1, self.total_numbers)
        return self._generate_combination_with_one_number(next_val)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _fallback(self) -> List[int]:
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())