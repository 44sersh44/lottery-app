import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class PositionalPredictor(BasePredictor):
    """Для каждой позиции вычисляет среднее, медиану, тренд и комбинирует."""

    def __init__(self, total_numbers: int, pick_count: int, name: str = "Positional"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.position_stats = []   # список для каждой позиции
        self.blocks = []           # сохраняем блоки для тренда
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        # Ожидаем, что blocks[0] — самый старый, но нам порядок не важен,
        # т.к. тренд мы вычисляем с учётом времени.
        # Сохраним blocks в исходном порядке (старый → новый) для тренда.
        self.blocks = blocks
        self.position_stats = []
        for pos in range(self.pick_count):
            values = [block[pos] for block in blocks if len(block) > pos]
            if not values:
                self.position_stats.append({'mean': 1, 'median': 1, 'trend': 1})
                continue
            mean_val = int(round(np.mean(values)))
            median_val = int(np.median(values))
            # Простой линейный тренд
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
            trend_val = int(round(values[-1] + slope))
            trend_val = max(1, min(self.total_numbers, trend_val))
            self.position_stats.append({
                'mean': mean_val,
                'median': median_val,
                'trend': trend_val
            })
        self.is_trained = True

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        return self.predict(1)[0]

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")

        predictions = []
        for offset in range(n_predictions):
            pred = []
            for pos, stats in enumerate(self.position_stats):
                # Комбинируем с небольшим случайным сдвигом для разных вариантов
                candidate = stats['mean']
                if offset % 2 == 0:
                    candidate = stats['median']
                if offset % 3 == 1:
                    candidate = stats['trend']
                # Не выходим за диапазон
                candidate = max(1, min(self.total_numbers, candidate))
                pred.append(candidate)
            # Убираем возможные дубликаты в пределах одного прогноза
            unique = []
            seen = set()
            for x in pred:
                if x not in seen:
                    unique.append(x)
                    seen.add(x)
            # Если после удаления дубликатов не хватает чисел, добиваем случайными
            while len(unique) < self.pick_count:
                new_num = np.random.randint(1, self.total_numbers + 1)
                if new_num not in seen:
                    unique.append(new_num)
                    seen.add(new_num)
            predictions.append(sorted(unique[:self.pick_count]))
        return predictions