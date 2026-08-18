import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

# Проверяем доступность sklearn
try:
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] sklearn не установлен, KNNPredictor будет использовать случайные прогнозы")


class KNNPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, n_neighbors: int = 5, name: str = "KNN"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.n_neighbors = n_neighbors
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not SKLEARN_OK or len(blocks) < 2:
            self.is_trained = True
            return
        X = np.array([sum(block) for block in blocks[:-1]]).reshape(-1, 1)
        y = [sum(block) for block in blocks[1:]]
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        if not SKLEARN_OK or self.model is None or not self.last_blocks:
            return self._fallback()
        last_sum = sum(self.last_blocks[-1])
        pred_sum = self.model.predict([[last_sum]])[0]
        return self._find_combo(pred_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """Возвращает несколько прогнозов."""
        return [self.predict_single() for _ in range(n_predictions)]

    def _find_combo(self, target_sum: float) -> List[int]:
        """Находит комбинацию чисел с суммой, близкой к target_sum."""
        for _ in range(2000):
            combo = sorted(np.random.choice(
                range(1, self.total_numbers + 1),
                self.pick_count,
                replace=False
            ).tolist())
            if abs(sum(combo) - target_sum) < 10:
                return combo
        return self._fallback()

    def _fallback(self) -> List[int]:
        """Случайный прогноз (если sklearn не доступен или нет данных)."""
        return sorted(np.random.choice(
            range(1, self.total_numbers + 1),
            self.pick_count,
            replace=False
        ).tolist())