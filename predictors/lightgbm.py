import numpy as np
from typing import List
from .base import BasePredictor

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class LightGBMPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, lookback: int = 3, name: str = "LightGBM"):
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.lookback = lookback
        self.model = None
        self.last_blocks = None
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not LGB_AVAILABLE or len(blocks) < self.lookback + 1:
            self.is_trained = False
            return
        X, y = [], []
        for i in range(len(blocks) - self.lookback):
            features = []
            for j in range(self.lookback):
                features.extend(blocks[i + j])
            X.append(features)
            y.append(sum(blocks[i + self.lookback]))
        X = np.array(X)
        y = np.array(y)
        self.model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not self.is_trained or self.model is None or self.last_blocks is None:
            return self._fallback()
        features = []
        for j in range(self.lookback):
            features.extend(self.last_blocks[-(self.lookback - j)])
        X_pred = np.array([features])
        pred_sum = self.model.predict(X_pred)[0]
        return self._find_combo_by_sum(pred_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _find_combo_by_sum(self, target_sum: float) -> List[int]:
        for _ in range(2000):
            combo = sorted(np.random.choice(range(1, self.total_numbers + 1),
                                            self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) < 10:
                return combo
        return self._fallback()

    def _fallback(self) -> List[int]:
        return sorted(np.random.choice(range(1, self.total_numbers + 1),
                                       self.pick_count, replace=False).tolist())