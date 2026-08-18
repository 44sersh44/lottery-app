import random
import numpy as np
from typing import List, Union, Tuple
from collections import Counter
from .base import BasePredictor


class MonteCarloPredictor(BasePredictor):
    """
    Монте-Карло предиктор.
    Проводит тысячи симуляций для оценки вероятностей и выбирает наиболее вероятные числа (детерминированно).
    """
    
    def __init__(self, total_numbers: int, pick_count: int, 
                 n_simulations: int = 10000, name: str = "MonteCarlo"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.n_simulations = n_simulations
        self.probabilities = None
        self.is_trained = False
        self.is_double = False
        self.total_numbers2 = 0
        self.pick_count2 = 0
    
    def fit(self, blocks: List[List[int]]) -> None:
        """Обучает модель на исторических данных."""
        if len(blocks) < 3:
            self.is_trained = False
            return
        
        # Считаем частоты
        freq = Counter()
        for block in blocks:
            for num in block:
                freq[num] += 1
        
        # Нормализуем
        total = sum(freq.values())
        if total == 0:
            self.probabilities = {i: 1/self.total_numbers for i in range(1, self.total_numbers+1)}
        else:
            self.probabilities = {num: count / total for num, count in freq.items()}
        
        self.is_trained = True
    
    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        """Обучение для двойной лотереи (только поле1)."""
        self.is_double = True
        self.total_numbers2 = max(max(b) for b in blocks2) if blocks2 else 54
        self.pick_count2 = len(blocks2[0]) if blocks2 and blocks2[0] else 1
        self.fit(blocks1)
    
    def _run_simulations(self) -> Counter:
        """Запускает симуляции и возвращает частоты."""
        numbers = list(self.probabilities.keys())
        weights = list(self.probabilities.values())
        
        results = Counter()
        
        for _ in range(self.n_simulations):
            chosen = set()
            while len(chosen) < self.pick_count:
                selected = random.choices(numbers, weights=weights)[0]
                chosen.add(selected)
            
            for num in chosen:
                results[num] += 1
        
        return results
    
    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        if not self.is_trained or self.probabilities is None:
            return self._fallback(n_predictions)
        
        # Запускаем симуляции
        sim_results = self._run_simulations()
        
        # Сортируем по частоте (убывание)
        sorted_nums = [num for num, _ in sim_results.most_common()]
        
        # Берем топ pick_count чисел (детерминированно)
        top_nums = sorted_nums[:self.pick_count]
        # Если не хватает чисел, добираем случайно (но это должно быть редко)
        while len(top_nums) < self.pick_count:
            available = set(range(1, self.total_numbers+1)) - set(top_nums)
            if available:
                # Добавляем случайное недостающее число (но для детерминизма лучше брать самое частое из оставшихся)
                # Для простоты возьмём следующее по частоте
                remaining = [num for num in sorted_nums if num not in top_nums]
                if remaining:
                    top_nums.append(remaining[0])
                else:
                    # Если нет оставшихся, просто берём случайное
                    top_nums.append(random.choice(list(available)))
            else:
                break
        top_nums.sort()
        
        # Для двойной лотереи возвращаем кортежи (поле1, поле2) – поле2 пока заглушка
        if self.is_double:
            # Для поля2 используем самое частое число из второго поля (если есть данные)
            # Но для простоты пока заглушка: середина диапазона
            p2 = [self.total_numbers2 // 2]
            return [(top_nums, p2)] * n_predictions
        else:
            return [top_nums] * n_predictions
    
    def predict_single(self) -> List[int]:
        return self.predict(1)[0]
    
    def _fallback(self, n_predictions: int) -> List[List[int]]:
        import random
        return [sorted(random.sample(range(1, self.total_numbers+1), self.pick_count)) 
                for _ in range(n_predictions)]