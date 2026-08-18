from .base import BasePredictor
from sklearn.ensemble import IsolationForest
import numpy as np
import random

class IsolationForestPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash(self.__class__.__name__) % 2**32
        self.model = None
        self.blocks = None

    def fit(self, blocks):
        rng = random.Random(self.random_state)
        if len(blocks) > 3:
            size = int(0.85 * len(blocks))
            indices = rng.choices(range(len(blocks)), k=size)
            sampled = [blocks[i] for i in indices]
        else:
            sampled = blocks
        self.blocks = sampled
        # Строим признаки: частоты чисел в каждом блоке
        X = []
        for b in sampled:
            freq = [0] * self.total_numbers
            for num in b:
                if 1 <= num <= self.total_numbers:
                    freq[num-1] += 1
            X.append(freq)
        X = np.array(X)
        if X.shape[0] < 2:
            self.model = None
            return self
        self.model = IsolationForest(random_state=self.random_state, contamination=0.1)
        self.model.fit(X)
        return self

    def predict_single(self):
        if self.model is None or not self.blocks:
            # fallback: случайный выбор
            rng = random.Random(self.random_state + 100)
            return sorted(rng.sample(range(1, self.total_numbers+1), self.pick_count))
        # Используем последний блок для предсказания
        last_block = self.blocks[-1]
        freq = [0] * self.total_numbers
        for num in last_block:
            if 1 <= num <= self.total_numbers:
                freq[num-1] += 1
        X_new = np.array([freq])
        scores = self.model.decision_function(X_new)[0]
        # Нормализуем оценки от 0 до 1
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            weights = np.ones(self.total_numbers)
        else:
            weights = (scores - min_score) / (max_score - min_score)
        # Выбираем top-k по весам
        numbers = list(range(1, self.total_numbers+1))
        sorted_nums = sorted(zip(numbers, weights), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:self.pick_count]])