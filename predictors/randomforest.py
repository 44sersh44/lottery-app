# randomforest.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

# Проверяем доступность sklearn
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARN] sklearn не установлен, RandomForestPredictor будет использовать случайные прогнозы")


class RandomForestPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, n_estimators: int = 50, name: str = "RandomForest"):
        # Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.n_estimators = n_estimators
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not HAS_SK or len(blocks) < 2:
            self.is_trained = True
            return
        X = np.array([sum(block) for block in blocks[:-1]]).reshape(-1, 1)
        y = np.array([sum(block) for block in blocks[1:]])
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not HAS_SK or self.model is None or len(self.last_blocks) == 0:
            return self._fallback(1)[0]
        last_sum = sum(self.last_blocks[-1])
        pred_sum = self.model.predict([[last_sum]])[0]
        return self._find_combo_by_sum(pred_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _find_combo_by_sum(self, target_sum: float, tolerance: int = 10) -> List[int]:
        for _ in range(2000):
            combo = sorted(np.random.choice(range(1, self.total_numbers + 1),
                                            self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) <= tolerance:
                return combo
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())

    def _fallback(self, n_predictions: int) -> List[List[int]]:
        return [sorted(np.random.choice(range(1, self.total_numbers + 1),
                                        self.pick_count, replace=False).tolist())
                for _ in range(n_predictions)]