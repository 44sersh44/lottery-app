import numpy as np
from typing import List
from .base import BasePredictor

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


class TransformerPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int,
                 d_model: int = 32, n_heads: int = 4,
                 num_layers: int = 2, **kwargs):
        super().__init__(total_numbers, pick_count)
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.lookback = 10
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not HAS_TENSORFLOW or len(blocks) < self.lookback + 1:
            self.is_trained = False
            return
        X, y = [], []
        for i in range(len(blocks) - self.lookback):
            X.append(blocks[i:i+self.lookback])
            y.append(blocks[i+self.lookback])
        X = np.array(X) / self.total_numbers
        y = np.array(y) / self.total_numbers
        inputs = keras.Input(shape=(self.lookback, self.pick_count))
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.pick_count)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=20, verbose=0)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not self.is_trained or self.model is None or len(self.last_blocks) < self.lookback:
            return []
        last = self.last_blocks[-self.lookback:]
        X = np.array([last]) / self.total_numbers
        pred = self.model.predict(X, verbose=0)[0] * self.total_numbers
        pred_int = np.round(pred).astype(int)
        pred_int = np.clip(pred_int, 1, self.total_numbers)
        unique = []
        for p in pred_int:
            if p not in unique:
                unique.append(p)
        while len(unique) < self.pick_count:
            for num in range(1, self.total_numbers+1):
                if num not in unique:
                    unique.append(num)
                    break
        return sorted(unique[:self.pick_count])

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]