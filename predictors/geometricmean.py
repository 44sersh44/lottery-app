"""Генетический алгоритм для подбора оптимальных комбинаций."""

import random
import numpy as np
from typing import List, Tuple, Union
from .base import BasePredictor
from collections import Counter


class GeneticPredictor(BasePredictor):
    """
    Генетический алгоритм для эволюционного подбора комбинаций.
    Использует частотный анализ как фитнес-функцию.
    """
    
    def __init__(self, total_numbers: int, pick_count: int, 
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 name: str = "Genetic"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.frequencies = None
    
    def fit(self, blocks: List[List[int]]) -> None:
        """Анализирует частоты чисел для фитнес-функции."""
        counter = Counter()
        for block in blocks:
            counter.update(block)
        
        max_freq = max(counter.values()) if counter else 1
        self.frequencies = {num: count / max_freq for num, count in counter.items()}
        
        for num in range(1, self.total_numbers + 1):
            if num not in self.frequencies:
                self.frequencies[num] = 0.1
        
        self.is_trained = True
    
    def _fitness(self, individual: List[int]) -> float:
        return sum(self.frequencies.get(num, 0) for num in individual)
    
    def _create_individual(self) -> List[int]:
        return sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        point = random.randint(1, self.pick_count - 1)
        child = list(set(parent1[:point] + parent2[point:]))
        while len(child) < self.pick_count:
            new_num = random.randint(1, self.total_numbers)
            if new_num not in child:
                child.append(new_num)
        while len(child) > self.pick_count:
            child.pop(random.randint(0, len(child)-1))
        return sorted(child)
    
    def _mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.pick_count - 1)
            new_num = random.randint(1, self.total_numbers)
            while new_num in individual:
                new_num = random.randint(1, self.total_numbers)
            individual[idx] = new_num
        return sorted(individual)
    
    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        if not self.is_trained:
            return self._fallback(n_predictions)
        
        population = [self._create_individual() for _ in range(self.population_size)]
        
        for _ in range(self.generations):
            fitness_scores = [(ind, self._fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            elite = [ind for ind, _ in fitness_scores[:self.population_size // 4]]
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                tournament = random.sample(fitness_scores[:self.population_size // 2], 2)
                parent1 = tournament[0][0]
                parent2 = tournament[1][0]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            population = new_population
        
        final_scores = [(ind, self._fitness(ind)) for ind in population]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        unique_predictions = []
        for ind, _ in final_scores:
            if ind not in unique_predictions:
                unique_predictions.append(ind)
        
        return unique_predictions[:n_predictions]
    
    def predict_single(self) -> List[int]:
        return self.predict(1)[0]
    
    def _fallback(self, n_predictions: int) -> List[List[int]]:
        import random
        return [sorted(random.sample(range(1, self.total_numbers+1), self.pick_count)) 
                for _ in range(n_predictions)]