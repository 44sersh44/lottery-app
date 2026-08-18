import numpy as np
from typing import List
from .base import BasePredictor

try:
    from sklearn.linear_model import Ridge
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class RidgePredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, alpha: float = 1.0, name: str = "Ridge"):
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.alpha = alpha
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not SKLEARN_OK or len(blocks) < 2:
            self.is_trained = True
            return
        X = np.array([sum(block) for block in blocks[:-1]]).reshape(-1, 1)
        y = np.array([sum(block) for block in blocks[1:]])
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        self.is_trained = True

    def _generate_combination_with_sum(self, target_sum: float, tolerance: int = 10) -> List[int]:
        if target_sum is None or not np.isfinite(target_sum):
            target_sum = 0
        for _ in range(2000):
            combo = sorted(np.random.choice(range(1, self.total_numbers + 1),
                                            self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) <= tolerance:
                return combo
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())

    def predict_single(self) -> List[int]:
        if not SKLEARN_OK or self.model is None or len(self.last_blocks) == 0:
            return self._generate_combination_with_sum(0, tolerance=999)
        last_sum = sum(self.last_blocks[-1])
        pred_sum = self.model.predict([[last_sum]])[0]
        return self._generate_combination_with_sum(pred_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]