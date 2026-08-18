"""
ARIMA Predictor для прогнозирования временных рядов номеров.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .base import BasePredictor

# Проверяем доступность statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[WARN] statsmodels не установлен, ARIMAPredictor будет использовать случайные прогнозы")


class ArimaPredictor(BasePredictor):
    """
    Прогнозирование на основе модели ARIMA для каждого числа.
    """
    def __init__(self, total_numbers: int, pick_count: int,
                 order: Tuple[int, int, int] = (5, 1, 0),
                 **kwargs):
        """
        :param total_numbers: Максимальное число в лотерее
        :param pick_count: Количество чисел в блоке
        :param order: Порядок ARIMA (p,d,q)
        """
        # ПРАВИЛЬНЫЙ вызов родительского конструктора
        super().__init__(total_numbers, pick_count)
        self.order = order
        self.models: Dict[int, object] = {}
        self.last_blocks: Optional[List[List[int]]] = None
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        """
        Обучает модель ARIMA для каждого числа на основе индексов его появления.
        """
        self.last_blocks = blocks
        if not STATSMODELS_AVAILABLE:
            self.is_trained = True
            return

        self.models.clear()
        # Собираем временные метки появления каждого числа
        data_by_number = {num: [] for num in range(1, self.total_numbers + 1)}
        for block_idx, block in enumerate(blocks):
            for num in block:
                data_by_number[num].append(block_idx)

        for num in range(1, self.total_numbers + 1):
            if data_by_number[num]:
                try:
                    model = ARIMA(data_by_number[num], order=self.order)
                    fitted = model.fit()
                    self.models[num] = fitted
                except Exception:
                    continue

        self.is_trained = True

    def predict_single(self) -> List[int]:
        """
        Возвращает один прогноз (список чисел).
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")

        if not STATSMODELS_AVAILABLE or not self.models:
            # fallback – случайный выбор
            return sorted(np.random.choice(
                range(1, self.total_numbers + 1),
                self.pick_count,
                replace=False
            ).tolist())

        forecasts = {}
        for num, model in self.models.items():
            try:
                forecast = model.forecast(steps=1)[0]
                forecasts[num] = forecast
            except Exception:
                pass

        if not forecasts:
            return sorted(np.random.choice(
                range(1, self.total_numbers + 1),
                self.pick_count,
                replace=False
            ).tolist())

        # Выбираем числа с наименьшим прогнозируемым интервалом (чем меньше, тем вероятнее)
        sorted_nums = sorted(forecasts.items(), key=lambda x: x[1])
        selected = [num for num, _ in sorted_nums[:self.pick_count]]
        return sorted(selected)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """
        Возвращает несколько прогнозов (для совместимости).
        """
        return [self.predict_single() for _ in range(n_predictions)]