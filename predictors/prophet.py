import numpy as np
import pandas as pd
from typing import List
from .base import BasePredictor

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ProphetPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int,
                 seasonality_mode: str = 'additive', **kwargs):
        super().__init__(total_numbers, pick_count)
        self.seasonality_mode = seasonality_mode
        self.models = []
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not HAS_PROPHET or len(blocks) < 10:
            self.is_trained = False
            return
        try:
            self.models = []
            dates = pd.date_range(start='2020-01-01', periods=len(blocks), freq='D')
            for pos in range(self.pick_count):
                df = pd.DataFrame({'ds': dates, 'y': [b[pos] for b in blocks]})
                model = Prophet(seasonality_mode=self.seasonality_mode)
                model.fit(df)
                self.models.append(model)
            self.is_trained = True
        except:
            self.is_trained = False

    def predict_single(self) -> List[int]:
        if not self.is_trained or not self.models:
            return []
        pred = []
        for model in self.models:
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            val = int(round(forecast['yhat'].iloc[-1]))
            val = max(1, min(val, self.total_numbers))
            pred.append(val)
        # Убираем дубликаты
        unique = []
        for p in pred:
            if p not in unique:
                unique.append(p)
        while len(unique) < self.pick_count:
            # Добавляем недостающие числа (ближайшие по значению)
            for num in range(1, self.total_numbers+1):
                if num not in unique:
                    unique.append(num)
                    break
        return sorted(unique[:self.pick_count])

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]