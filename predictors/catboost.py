import numpy as np
import random
from typing import List
from .base import BasePredictor

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class CatBoostPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, lookback: int = 3, name: str = "CatBoost"):
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.lookback = lookback
        self.model = None
        self.last_blocks = None
        self.is_trained = False
        self._rng = random.Random(hash(name) & 0xFFFFFFFF)

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not CATBOOST_AVAILABLE or len(blocks) < self.lookback + 1:
            self.is_trained = False
            return
        X, y = [], []
        for i in range(len(blocks) - self.lookback):
            feat = []
            for j in range(self.lookback):
                feat.extend(blocks[i + j])
            X.append(feat)
            y.append(sum(blocks[i + self.lookback]))
        self.model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=False)
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if self.is_trained and self.model is not None and self.last_blocks is not None:
            feat = []
            for j in range(self.lookback):
                feat.extend(self.last_blocks[-(self.lookback - j)])
            pred_sum = self.model.predict([feat])[0]
            return self._find_combo(pred_sum)
        return sorted(self._rng.sample(range(1, self.total_numbers + 1), self.pick_count))

    def _find_combo(self, target: float) -> List[int]:
        for _ in range(2000):
            combo = sorted(self._rng.sample(range(1, self.total_numbers + 1), self.pick_count))
            if abs(sum(combo) - target) < 10:
                return combo
        return sorted(self._rng.sample(range(1, self.total_numbers + 1), self.pick_count))

    def predict(self, n: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n)]