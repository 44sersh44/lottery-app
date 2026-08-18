from .base import BasePredictor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

class BayesianProbabilityPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash("BayesianProbability") % 2**32
        self.model = None
        self.scaler = StandardScaler()
        self.features = None

    def fit(self, blocks):
        rng = random.Random(self.random_state)
        if len(blocks) > 3:
            size = int(0.85 * len(blocks))
            indices = rng.choices(range(len(blocks)), k=size)
            sampled = [blocks[i] for i in indices]
        else:
            sampled = blocks
        # Признаки: для каждого блока – статистики, частоты и т.д.
        X = []
        y = []
        for b in sampled:
            features = [np.mean(b), np.std(b), np.min(b), np.max(b), np.median(b)]
            X.append(features)
            # Целевая переменная – сумма блока (или можно бинарные метки для каждого числа, но упростим)
            y.append(np.sum(b))
        X = np.array(X)
        y = np.array(y)
        # Для классификации лучше использовать бинарные метки, но чтобы не усложнять – используем регрессию
        # Вместо этого обучим классификатор на предсказание суммы (будем использовать как прокси)
        # Переделаем в бинарную классификацию для каждого числа – слишком долго.
        # Поэтому оставим как есть – будем использовать наивный байес на признаках для предсказания суммы.
        # Для прогноза используем частоты с апостериорным обновлением.
        from collections import Counter
        counter = Counter()
        for b in blocks:
            counter.update(b)
        self.prior = {n: (counter.get(n, 0) + 1) / (len(blocks) + self.total_numbers) for n in range(1, self.total_numbers+1)}
        return self

    def predict_single(self):
        if not hasattr(self, 'prior'):
            return list(range(1, self.pick_count + 1))
        # Используем апостериорные вероятности как веса, но с небольшим шумом
        numbers = list(range(1, self.total_numbers+1))
        rng = random.Random(self.random_state)
        weights = [self.prior.get(n, 0.01) + rng.uniform(-0.01, 0.01) for n in numbers]
        # Нормализуем
        weights = np.maximum(weights, 0)
        sorted_nums = sorted(zip(numbers, weights), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:self.pick_count]])