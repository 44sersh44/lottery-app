from .base import BasePredictor
from collections import Counter
import random
import math

class PoissonPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash(self.__class__.__name__) % 2**32
        self.lambdas = None
        self.temperature = 0.3

    def fit(self, blocks):
        if not hasattr(self, 'temperature'):
            self.temperature = 0.3
        rng = random.Random(self.random_state)
        if len(blocks) > 5:
            size = int(0.85 * len(blocks))
            indices = rng.choices(range(len(blocks)), k=size)
            sampled = [blocks[i] for i in indices]
        else:
            sampled = blocks
        counter = Counter()
        for b in sampled:
            counter.update(b)
        self.lambdas = {n: counter.get(n, 0) + 0.1 for n in range(1, self.total_numbers+1)}
        return self

    def predict_single(self):
        if not hasattr(self, 'temperature'):
            self.temperature = 0.3
        if self.lambdas is None:
            return list(range(1, self.pick_count + 1))

        numbers = list(range(1, self.total_numbers + 1))
        weights = [self.lambdas[n] for n in numbers]
        # Добавляем небольшой шум, чтобы избежать нулевых вероятностей
        weights = [w + 1e-9 for w in weights]

        # Берём топ-N (N = pick_count * 3) для вероятностного выбора
        top_n = min(self.pick_count * 3, self.total_numbers)
        sorted_idx = sorted(range(len(numbers)), key=lambda i: weights[i], reverse=True)[:top_n]
        top_numbers = [numbers[i] for i in sorted_idx]
        top_weights = [weights[i] for i in sorted_idx]

        # Softmax с температурой
        max_w = max(top_weights)
        exp_w = [math.exp((w - max_w) / self.temperature) for w in top_weights]
        probs = [w / sum(exp_w) for w in exp_w]

        rng = random.Random(self.random_state + 2)

        # Выбираем pick_count чисел без повторений (семплирование без возвращения)
        selected = []
        # Сначала делаем выборку с весами, но без повторений (имитация без возврата)
        # Для этого используем метод choices с replace=False – но он не поддерживает веса.
        # Поэтому делаем так: многократно выбираем, удаляя выбранное.
        temp_numbers = top_numbers[:]
        temp_probs = probs[:]
        for _ in range(self.pick_count):
            if not temp_numbers:
                break
            # Нормализуем вероятности для оставшихся
            total_prob = sum(temp_probs)
            if total_prob == 0:
                # Если все вероятности нулевые, берём случайное число
                choice = rng.choice(temp_numbers)
            else:
                norm_probs = [p / total_prob for p in temp_probs]
                choice = rng.choices(temp_numbers, weights=norm_probs, k=1)[0]
            selected.append(choice)
            # Удаляем выбранное число и его вероятность
            idx = temp_numbers.index(choice)
            del temp_numbers[idx]
            del temp_probs[idx]

        # Если по какой-то причине не набрали нужное количество, добираем случайными
        while len(selected) < self.pick_count:
            available = [n for n in numbers if n not in selected]
            if not available:
                break
            new_num = rng.choice(available)
            selected.append(new_num)

        # Обрезаем до pick_count (на всякий случай)
        selected = selected[:self.pick_count]
        return sorted(selected)