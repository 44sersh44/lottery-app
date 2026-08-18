from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from .base import BasePredictor

# Проверяем доступность sklearn
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class FrequencyPredictor(BasePredictor):
    def __init__(self, total_numbers, pick_count, **kwargs):
        super().__init__(total_numbers, pick_count)   # <-- ВАЖНО: вызываем super()
        # ... свои параметры


class _BaseEnsemblePredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "BaseEnsemble", **model_kwargs):
        # Исправленный вызов super: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.model_kwargs = model_kwargs
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
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_trained = True

    def _create_model(self):
        raise NotImplementedError

    def _find_combo(self, target_sum):
        for _ in range(2000):
            combo = sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())
            if abs(sum(combo) - target_sum) < 10:
                return combo
        return sorted(np.random.choice(range(1, self.total_numbers+1), self.pick_count, replace=False).tolist())

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        if not SKLEARN_OK or self.model is None or not self.last_blocks:
            return [self._find_combo(0) for _ in range(n_predictions)]
        last_sum = sum(self.last_blocks[-1])
        pred_sum = self.model.predict([[last_sum]])[0]
        return [self._find_combo(pred_sum) for _ in range(n_predictions)]

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз (для совместимости с BasePredictor)."""
        return self.predict(1)[0]