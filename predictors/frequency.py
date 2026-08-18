from .base import BasePredictor
from collections import Counter
import random

class FrequencyPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash("Frequency") % 2**32
        self.freq = None

    def fit(self, blocks):
        rng = random.Random(self.random_state)
        if len(blocks) > 3:
            size = int(0.85 * len(blocks))
            indices = rng.choices(range(len(blocks)), k=size)
            sampled = [blocks[i] for i in indices]
        else:
            sampled = blocks
        counter = Counter()
        for b in sampled:
            counter.update(b)
        self.freq = counter
        return self

    def predict_single(self):
        if self.freq is None:
            return list(range(1, self.pick_count + 1))
        sorted_nums = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:self.pick_count]])