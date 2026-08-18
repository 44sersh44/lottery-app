import numpy as np
import random
from typing import List, Dict, Tuple, Union
from collections import defaultdict, Counter
from .base import BasePredictor

class MarkovPredictor(BasePredictor):
    """
    Марковская цепь для прогнозирования следующих чисел.
    Учитывает порядок (order) и сглаживание для редких состояний.
    """
    
    def __init__(self, total_numbers: int, pick_count: int, order: int = 1, name: str = "Markov"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.order = order
        self.transitions = defaultdict(Counter)
        self.start_states = Counter()
        self.is_trained = False
    
    def fit(self, blocks: List[List[int]]) -> None:
        """Обучает марковскую цепь."""
        if len(blocks) < self.order + 1:
            print(f"  [WARN] Markov: нужно минимум {self.order + 1} блоков, есть {len(blocks)}")
            self.is_trained = False
            return
        
        self.transitions.clear()
        self.start_states.clear()
        
        for block in blocks:
            # Убеждаемся, что блок отсортирован
            sorted_block = sorted(block)
            
            # Запоминаем начальные состояния
            if len(sorted_block) >= self.order:
                start_state = tuple(sorted_block[:self.order])
                self.start_states[start_state] += 1
            
            # Строим переходы
            for i in range(len(sorted_block) - self.order):
                state = tuple(sorted_block[i:i+self.order])
                next_num = sorted_block[i+self.order]
                self.transitions[state][next_num] += 1
        
        # Нормализуем вероятности и добавляем сглаживание
        for state in self.transitions:
            total = sum(self.transitions[state].values())
            if total > 0:
                for num in list(self.transitions[state].keys()):
                    self.transitions[state][num] = self.transitions[state][num] / total
        
        # Добавляем сглаживание для всех возможных чисел
        for state in list(self.transitions.keys()):
            for num in range(1, self.total_numbers + 1):
                if num not in self.transitions[state]:
                    self.transitions[state][num] = 0.01
        
        # Если нет начальных состояний, создаём их
        if not self.start_states:
            freq = Counter()
            for block in blocks:
                for num in block:
                    freq[num] += 1
            
            top_numbers = [num for num, _ in freq.most_common(self.order * 2)]
            for i in range(len(top_numbers) - self.order + 1):
                if i + self.order <= len(top_numbers):
                    state = tuple(top_numbers[i:i+self.order])
                    self.start_states[state] = freq.get(state[0], 1)
        
        self.is_trained = True
        print(f"  [OK] Markov: {len(self.transitions)} состояний, {len(self.start_states)} начальных")
    
    def _get_next_number(self, state: Tuple[int, ...]) -> int:
        """Предсказывает следующее число на основе состояния."""
        if state not in self.transitions:
            return random.randint(1, self.total_numbers)
        
        probs = self.transitions[state]
        numbers = list(probs.keys())
        weights = list(probs.values())
        
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return np.random.choice(numbers, p=weights)
    
    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """Предсказывает следующие блоки."""
        if not self.is_trained:
            return self._fallback(n_predictions)
        
        predictions = []
        
        for _ in range(n_predictions):
            # Выбираем начальное состояние
            if self.start_states:
                states = list(self.start_states.keys())
                weights = list(self.start_states.values())
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    start_state = list(states[np.random.choice(len(states), p=weights)])
                else:
                    start_state = list(random.choice(states))
            else:
                start_state = [random.randint(1, self.total_numbers) for _ in range(min(self.order, self.pick_count))]
                start_state.sort()
            
            # Генерируем последовательность
            result = start_state.copy()
            
            while len(result) < self.pick_count:
                # Берём последние order чисел
                if len(result) >= self.order:
                    state = tuple(result[-self.order:])
                else:
                    state = tuple(result)
                next_num = self._get_next_number(state)
                if next_num not in result:
                    result.append(next_num)
                result.sort()
            
            # Обрезаем до pick_count
            result = result[:self.pick_count]
            predictions.append(result)
        
        return predictions
    
    def predict_single(self) -> List[int]:
        return self.predict(1)[0]
    
    def _fallback(self, n_predictions: int) -> List[List[int]]:
        """Заглушка - случайные числа."""
        import random
        return [sorted(random.sample(range(1, self.total_numbers+1), self.pick_count)) 
                for _ in range(n_predictions)]