import numpy as np
from typing import List
from .base import BasePredictor

try:
    from sklearn.naive_bayes import BernoulliNB
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class BernoulliNBPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, **kwargs):
        super().__init__(total_numbers, pick_count)
        self.model = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not SKLEARN_OK or len(blocks) < 2:
            self.is_trained = False
            return
        X = np.zeros((len(blocks), self.total_numbers))
        for i, b in enumerate(blocks):
            for num in b:
                X[i, num - 1] = 1
        sums = [sum(b) for b in blocks]
        bins = np.percentile(sums, [33, 66])
        y = np.digitize(sums, bins)
        self.model = BernoulliNB()
        self.model.fit(X, y)
        self.is_trained = True

    def predict_single(self) -> List[int]:
        if not self.is_trained or self.model is None or not self.last_blocks:
            return []
        last_vec = np.zeros(self.total_numbers)
        for num in self.last_blocks[-1]:
            last_vec[num - 1] = 1
        cls = self.model.predict([last_vec])[0]
        sums = [sum(b) for b in self.last_blocks]
        bins = np.percentile(sums, [33, 66])
        if cls == 0:
            target = bins[0] / 2
        elif cls == 1:
            target = (bins[0] + bins[1]) / 2
        else:
            target = bins[1] + (sums[-1] - bins[1]) / 2
        # Находим комбинацию чисел, сумма которых близка к target
        best_combo = []
        best_diff = float('inf')
        # Генерируем все комбинации, но для скорости используем эвристику: берём числа из частот
        from collections import Counter
        freq_counter = Counter()
        for b in self.last_blocks:
            freq_counter.update(b)
        top_nums = [num for num, _ in freq_counter.most_common(self.total_numbers)]
        # Ищем комбинацию с ближайшей суммой (жадный алгоритм)
        for _ in range(1000):
            combo = sorted(np.random.choice(top_nums, self.pick_count, replace=False).tolist())
            diff = abs(sum(combo) - target)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo
            if diff == 0:
                break
        return sorted(best_combo) if best_combo else []

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]