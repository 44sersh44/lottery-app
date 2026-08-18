import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class KNNPatternPredictor(BasePredictor):
    """KNN, который смотрит на паттерны разностей между числами в блоке."""
    def __init__(self, total_numbers: int, pick_count: int, n_neighbors: int = 3, name: str = "KNNPattern"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.n_neighbors = n_neighbors
        self.patterns = []
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        # Сохраняем паттерны (разности соседних чисел) для каждого блока
        self.patterns = []
        for block in blocks:
            diffs = [block[i+1] - block[i] for i in range(len(block)-1)]
            self.patterns.append(diffs)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        if not self.patterns or not self.last_blocks:
            return self._fallback()
        # Находим последний блок и ищем похожие паттерны
        last_block = self.last_blocks[-1]
        last_diffs = [last_block[i+1] - last_block[i] for i in range(len(last_block)-1)]
        # Вычисляем расстояния (сумма модулей разностей)
        distances = [sum(abs(d - ld) for d, ld in zip(pattern, last_diffs)) for pattern in self.patterns]
        nearest_idx = np.argsort(distances)[:self.n_neighbors]
        # Берём следующий блок из ближайших соседей (если есть)
        candidates = [self.last_blocks[i+1] for i in nearest_idx if i+1 < len(self.last_blocks)]
        if not candidates:
            return self._fallback()
        chosen = candidates[np.random.randint(len(candidates))]
        return sorted(chosen)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """Возвращает несколько прогнозов."""
        return [self.predict_single() for _ in range(n_predictions)]

    def _fallback(self) -> List[int]:
        """Случайный прогноз (если нет данных)."""
        return sorted(np.random.choice(
            range(1, self.total_numbers + 1),
            self.pick_count,
            replace=False
        ).tolist())