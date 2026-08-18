from .base import BasePredictor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from core.features import extract_features
import numpy as np
import random

class MLP(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None,
                 hidden_layer_sizes=(100, 50), activation='relu',
                 alpha=0.001, max_iter=500, random_state=42, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state + 1  # уникальный seed
        self.model = None
        self.scaler = StandardScaler()
        self.blocks = None

    def fit(self, blocks):
        self.blocks = blocks
        X, y = extract_features(blocks, self.total_numbers)
        if X.shape[0] == 0:
            return self
        # Обучаем многоклассовый классификатор (один на все числа)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False
        )
        X_scaled = self.scaler.fit_transform(X)
        # Обучаем на бинарных метках (multi-label) – используем OneVsRestClassifier
        from sklearn.multiclass import OneVsRestClassifier
        self.model = OneVsRestClassifier(
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False
            )
        )
        self.model.fit(X_scaled, y)
        return self

    def predict_single(self):
        if self.blocks is None or len(self.blocks) == 0:
            # fallback
            return list(range(1, self.pick_count+1))
        # Берём последний блок для предсказания
        last_block = self.blocks[-1]
        X_new, _ = extract_features([last_block], self.total_numbers)
        X_new_scaled = self.scaler.transform(X_new)
        proba = self.model.predict_proba(X_new_scaled)[0]
        # proba – массив (n_classes, )? при OneVsRest выдаёт список массивов.
        # Проще: берём средние вероятности по всем классам.
        if isinstance(proba, list):
            proba = np.mean(proba, axis=0)
        # Выбираем top-k чисел с наибольшей вероятностью
        top_indices = np.argsort(proba)[-self.pick_count:][::-1]
        result = [i+1 for i in top_indices]
        return sorted(result)