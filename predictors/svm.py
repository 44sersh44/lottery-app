# svm.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

try:
    from sklearn.svm import SVR
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] sklearn не установлен, SVMPredictor будет использовать случайные прогнозы")


class SVMPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', name: str = "SVM"):
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
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
        self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not SKLEARN_OK or self.model is None or not self.last_blocks:
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