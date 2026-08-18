# xgboost.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost не установлен, XGBoostPredictor будет использовать случайные прогнозы")


class XGBoostPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, n_estimators: int = 100, 
                 learning_rate: float = 0.1, max_depth: int = 3, name: str = "XGBoost"):
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not HAS_XGB or len(blocks) < 2:
            self.is_trained = True
            return
        X = np.array([sum(block) for block in blocks[:-1]]).reshape(-1, 1)
        y = np.array([sum(block) for block in blocks[1:]])
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      max_depth=self.max_depth, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not HAS_XGB or self.model is None or not self.last_blocks:
            return self._fallback()
        last_sum = sum(self.last_blocks[-1])
        pred_sum = self.model.predict([[last_sum]])[0]
        return self._find_combo(pred_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _find_combo(self, target_sum: float) -> List[int]:
        for _ in range(2000):
            combo = sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) < 10:
                return combo
        return self._fallback()

    def _fallback(self) -> List[int]:
        return sorted(np.random.choice(range(1, self.total_numbers + 1), self.pick_count, replace=False).tolist())