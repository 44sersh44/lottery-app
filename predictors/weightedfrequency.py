from .base import BasePredictor
import random

class WeightedFrequencyPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash("WeightedFrequency") % 2**32
        # Разный decay для каждой модели
        rng = random.Random(self.random_state)
        self.decay = rng.uniform(0.7, 0.99)
        self.weights = None

    def fit(self, blocks):
        rng = random.Random(self.random_state)
        if len(blocks) > 3:
            size = int(0.85 * len(blocks))
            indices = rng.choices(range(len(blocks)), k=size)
            sampled = [blocks[i] for i in indices]
        else:
            sampled = blocks
        weighted = {}
        for i, b in enumerate(sampled):
            weight = self.decay ** (len(sampled) - i - 1)
            for num in b:
                weighted[num] = weighted.get(num, 0) + weight
        self.weights = {n: weighted.get(n, 0) for n in range(1, self.total_numbers+1)}
        return self

    def predict_single(self):
        if self.weights is None:
            return list(range(1, self.pick_count + 1))
        sorted_nums = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:self.pick_count]])