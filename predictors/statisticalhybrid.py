import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor


class StatisticalHybridPredictor(BasePredictor):
    """
    Предиктор, использующий статистический анализ:
    - чётность
    - распределение по десяткам
    - частотность чисел
    - холодные/горячие числа
    """
    
    def __init__(self, total_numbers: int, pick_count: int, name: str = "StatisticalHybrid"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.frequency = None
        self.cold_numbers = []
        self.hot_numbers = []
        self.even_ratio = 0.5
        self.decade_weights = [0.25, 0.25, 0.25, 0.25]
        self.is_trained = False
    
    def fit(self, blocks: List[List[int]]) -> None:
        from collections import Counter
        
        # Частотный анализ
        counter = Counter()
        for block in blocks:
            counter.update(block)
        
        # Горячие числа (топ-10)
        self.hot_numbers = [num for num, _ in counter.most_common(10)]
        
        # Холодные числа (нижние 10)
        all_nums = set(range(1, self.total_numbers + 1))
        freq_nums = set(counter.keys())
        cold_candidates = list(all_nums - freq_nums)
        cold_candidates.extend([num for num, count in counter.items() if count <= 2])
        self.cold_numbers = list(set(cold_candidates))[:10]
        
        # Анализ чётности
        even_count = sum(1 for num in counter.elements() if num % 2 == 0)
        odd_count = sum(1 for num in counter.elements() if num % 2 == 1)
        total = even_count + odd_count
        self.even_ratio = even_count / total if total > 0 else 0.5
        
        # Анализ десятков
        self.decade_weights = [0, 0, 0, 0]
        for num in counter.elements():
            if num <= 9: self.decade_weights[0] += 1
            elif num <= 19: self.decade_weights[1] += 1
            elif num <= 29: self.decade_weights[2] += 1
            else: self.decade_weights[3] += 1
        
        total_dec = sum(self.decade_weights)
        if total_dec > 0:
            self.decade_weights = [w / total_dec for w in self.decade_weights]
        else:
            self.decade_weights = [0.25, 0.25, 0.25, 0.25]
        
        self.is_trained = True
    
    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        return self._generate_prediction()
    
    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """Возвращает несколько прогнозов."""
        return [self._generate_prediction() for _ in range(n_predictions)]
    
    def _generate_prediction(self) -> List[int]:
        result = []
        
        # 1. Берём 3-4 горячих числа
        hot_count = np.random.choice([3, 4], p=[0.6, 0.4])
        if len(self.hot_numbers) >= hot_count:
            hot_selected = np.random.choice(self.hot_numbers, size=hot_count, replace=False).tolist()
            result.extend(hot_selected)
        
        # 2. Берём 2-3 холодных числа
        cold_count = self.pick_count - len(result)
        cold_count = min(cold_count, len(self.cold_numbers))
        if cold_count > 0 and len(self.cold_numbers) >= cold_count:
            cold_selected = np.random.choice(self.cold_numbers, size=cold_count, replace=False).tolist()
            result.extend(cold_selected)
        
        # 3. Добираем до pick_count
        while len(result) < self.pick_count:
            # Выбираем на основе весов десятков
            decade = np.random.choice([0, 1, 2, 3], p=self.decade_weights)
            if decade == 0:
                nums = list(range(1, 10))
            elif decade == 1:
                nums = list(range(10, 20))
            elif decade == 2:
                nums = list(range(20, 30))
            else:
                nums = list(range(30, self.total_numbers + 1))
            
            available = [n for n in nums if n not in result]
            if available:
                result.append(np.random.choice(available))
            else:
                # fallback
                all_available = [n for n in range(1, self.total_numbers + 1) if n not in result]
                if all_available:
                    result.append(np.random.choice(all_available))
                else:
                    # если все числа уже использованы (невозможно)
                    break
        
        return sorted(result)