import numpy as np
from typing import List
from .base import BasePredictor

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, seq_length: int = 3, **kwargs):
        super().__init__(total_numbers, pick_count)
        self.seq_length = seq_length
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not TF_AVAILABLE or len(blocks) < self.seq_length + 1:
            self.is_trained = False
            return
        X, y = [], []
        for i in range(len(blocks) - self.seq_length):
            X.append(blocks[i:i+self.seq_length])
            y.append(blocks[i+self.seq_length])
        X = np.array(X) / self.total_numbers
        y = np.array(y) / self.total_numbers
        self.model = Sequential([
            LSTM(50, input_shape=(self.seq_length, self.pick_count)),
            Dense(self.pick_count, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=20, verbose=0)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not self.is_trained or self.model is None or len(self.last_blocks) < self.seq_length:
            return []
        last = self.last_blocks[-self.seq_length:]
        X = np.array([last]) / self.total_numbers
        pred = self.model.predict(X, verbose=0)[0] * self.total_numbers
        pred_int = np.round(pred).astype(int)
        pred_int = np.clip(pred_int, 1, self.total_numbers)
        # Убираем дубликаты
        unique = []
        for p in pred_int:
            if p not in unique:
                unique.append(p)
        # Если не хватает чисел, добавляем недостающие (но без случайности, просто берём ближайшие)
        while len(unique) < self.pick_count:
            # Добавляем число, которое не входит, но ближайшее по значению
            for num in range(1, self.total_numbers+1):
                if num not in unique:
                    unique.append(num)
                    break
        return sorted(unique[:self.pick_count])

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]