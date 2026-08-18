import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class LinearRegressionPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "LinearReg"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.coefs = []
        self.intercepts = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        # blocks: от старых к новым, ожидаем список блоков
        if len(blocks) < 2:
            self.coefs = [0] * self.pick_count
            self.intercepts = [0] * self.pick_count
            self.is_trained = True
            return
        self.coefs = []
        self.intercepts = []
        x = np.arange(len(blocks)).reshape(-1, 1)
        for pos in range(self.pick_count):
            values = [block[pos] for block in blocks]
            y = np.array(values)
            if len(y) > 1:
                coef = np.polyfit(x.flatten(), y, 1)[0]
                intercept = np.polyfit(x.flatten(), y, 1)[1]
            else:
                coef, intercept = 0, y[0] if y else 1
            self.coefs.append(coef)
            self.intercepts.append(intercept)
        self.is_trained = True

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        predictions = []
        for offset in range(n_predictions):
            next_idx = len(self.coefs) + offset  # будущий индекс (смещение)
            pred = []
            for pos in range(self.pick_count):
                val = self.coefs[pos] * next_idx + self.intercepts[pos]
                val = int(round(val))
                val = max(1, min(self.total_numbers, val))
                pred.append(val)
            # убираем возможные дубликаты
            unique = []
            for x in pred:
                if x not in unique:
                    unique.append(x)
            while len(unique) < self.pick_count:
                new_num = np.random.randint(1, self.total_numbers+1)
                if new_num not in unique:
                    unique.append(new_num)
            predictions.append(sorted(unique[:self.pick_count]))
        return predictions

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз (для совместимости с BasePredictor)."""
        return self.predict(1)[0]