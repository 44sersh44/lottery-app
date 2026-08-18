import numpy as np
import statistics
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class MedianPredictor(BasePredictor):
    def __init__(self, total_numbers: int, pick_count: int, name: str = "Median"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.median_nums = None
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
        self.median_nums2 = None
        self.blocks = []

    def fit(self, blocks: List[List[int]]) -> None:
        self.blocks = blocks
        if not blocks:
            self.median_nums = list(range(1, min(self.pick_count, self.total_numbers) + 1))
        else:
            pos_values = [[] for _ in range(self.pick_count)]
            for block in blocks:
                sorted_block = sorted(block)
                for pos, num in enumerate(sorted_block):
                    if pos < self.pick_count:
                        pos_values[pos].append(num)
            self.median_nums = [int(np.median(v)) for v in pos_values if v]
            if len(self.median_nums) < self.pick_count:
                self.median_nums.extend(range(len(self.median_nums) + 1, self.pick_count + 1))
        self.is_trained = True

    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        self.is_double = True
        self.blocks2 = blocks2
        self.fit(blocks1)
        if blocks2 and self.pick_count2 > 0:
            pos_values2 = [[] for _ in range(self.pick_count2)]
            for block in blocks2:
                for pos, num in enumerate(sorted(block)):
                    if pos < self.pick_count2:
                        pos_values2[pos].append(num)
            self.median_nums2 = [int(np.median(v)) for v in pos_values2 if v]
            if len(self.median_nums2) < self.pick_count2:
                self.median_nums2.extend(range(len(self.median_nums2) + 1, self.pick_count2 + 1))

    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if self.is_double:
            return self._predict_double(n_predictions)
        return self._predict_single(n_predictions)

    def predict_single(self) -> List[int]:
        if not self.blocks:
            return self._fallback_single(1)[0]
        result = []
        for pos in range(self.pick_count):
            values = [block[pos] for block in self.blocks if len(block) > pos]
            if values:
                median_val = int(statistics.median(values))
                result.append(median_val)
            else:
                result.append(random.randint(1, self.total_numbers))
        # Убираем дубликаты
        result = sorted(set(result))
        while len(result) < self.pick_count:
            new_num = random.randint(1, self.total_numbers)
            if new_num not in result:
                result.append(new_num)
        return result[:self.pick_count]

    def _predict_single(self, n_predictions: int) -> List[List[int]]:
        return [self.predict_single() for _ in range(n_predictions)]

    def _predict_double(self, n_predictions: int) -> List[Tuple[List[int], List[int]]]:
        preds1 = self._predict_single(n_predictions)
        if self.median_nums2:
            base2 = sorted(self.median_nums2)
            preds2 = [[base2[0]] for _ in range(n_predictions)]
        else:
            preds2 = [[self.total_numbers2 // 2] for _ in range(n_predictions)]
        return [(preds1[i], preds2[i]) for i in range(n_predictions)]

    def _fallback_single(self, n_predictions: int) -> List[List[int]]:
        return [sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
                for _ in range(n_predictions)]