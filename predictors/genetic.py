from .base import BasePredictor
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from core.features import extract_features

class GeneticPredictor(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None, random_state=None, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.random_state = random_state or hash(self.__class__.__name__ + "_regression") % 2**32
        self.model = None
        self.scaler = StandardScaler()
        self.blocks = None
        self.best_alpha = 1.0
        self.best_tol = 1e-4

    def _genetic_optimize(self, X, y):
        """Генетический подбор параметров alpha и tol для Ridge."""
        # Ген – (alpha, tol)
        pop_size = 20
        generations = 10
        mutation_rate = 0.2
        
        # Начальная популяция (логарифмическое пространство)
        rng = random.Random(self.random_state + 999)
        population = []
        for _ in range(pop_size):
            alpha = 10 ** rng.uniform(-4, 4)
            tol = 10 ** rng.uniform(-6, -2)
            population.append((alpha, tol))
        
        # Функция приспособленности (средняя точность на всей выборке)
        def fitness(alpha, tol):
            try:
                model = Ridge(alpha=alpha, tol=tol, random_state=self.random_state)
                model.fit(X, y)
                # Оценка: средняя абсолютная ошибка на обучающей выборке (чем меньше, тем лучше)
                pred = model.predict(X)
                # Для бинарных меток считаем среднеквадратичную ошибку
                mse = np.mean((pred - y) ** 2)
                return -mse  # минимизируем MSE
            except:
                return -1e9
        
        for gen in range(generations):
            # Оценка
            scores = [fitness(alpha, tol) for alpha, tol in population]
            # Отбор лучших (турнирный)
            new_pop = []
            for _ in range(pop_size // 2):
                # Турнирный отбор (два случайных, выбираем лучшего)
                idx1 = rng.randint(0, pop_size-1)
                idx2 = rng.randint(0, pop_size-1)
                if scores[idx1] > scores[idx2]:
                    winner = population[idx1]
                else:
                    winner = population[idx2]
                new_pop.append(winner)
            # Кроссинговер (усреднение)
            while len(new_pop) < pop_size:
                p1 = rng.choice(new_pop)
                p2 = rng.choice(new_pop)
                child_alpha = (p1[0] + p2[0]) / 2 * (1 + rng.uniform(-0.1, 0.1))
                child_tol = (p1[1] + p2[1]) / 2 * (1 + rng.uniform(-0.1, 0.1))
                child_alpha = max(1e-6, child_alpha)
                child_tol = max(1e-8, child_tol)
                new_pop.append((child_alpha, child_tol))
            # Мутация
            for i in range(len(new_pop)):
                if rng.random() < mutation_rate:
                    alpha, tol = new_pop[i]
                    if rng.random() < 0.5:
                        alpha *= 10 ** rng.uniform(-0.5, 0.5)
                    else:
                        tol *= 10 ** rng.uniform(-0.5, 0.5)
                    alpha = max(1e-6, alpha)
                    tol = max(1e-8, tol)
                    new_pop[i] = (alpha, tol)
            population = new_pop
        
        # Лучший
        scores = [fitness(alpha, tol) for alpha, tol in population]
        best_idx = np.argmax(scores)
        best_alpha, best_tol = population[best_idx]
        # Дополнительная проверка: если все оценки плохие, берём дефолтные
        if scores[best_idx] < -1e8:
            best_alpha, best_tol = 1.0, 1e-4
        return best_alpha, best_tol

    def fit(self, blocks):
        self.blocks = blocks
        if len(blocks) < 3:
            return self
        
        # Извлекаем признаки и метки
        X, y = extract_features(blocks, self.total_numbers)
        if X.shape[0] == 0:
            return self
        
        # Генетический подбор параметров
        self.best_alpha, self.best_tol = self._genetic_optimize(X, y)
        
        # Обучаем финальную модель с лучшими параметрами
        self.model = Ridge(alpha=self.best_alpha, tol=self.best_tol, random_state=self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_single(self):
        if self.model is None or self.blocks is None or len(self.blocks) == 0:
            return list(range(1, self.pick_count + 1))
        
        # Берём последний блок для предсказания следующего
        last_block = self.blocks[-1]
        X_new, _ = extract_features([last_block], self.total_numbers)
        if X_new.shape[0] == 0:
            return list(range(1, self.pick_count + 1))
        
        X_new_scaled = self.scaler.transform(X_new)
        # Предсказание вероятностей для каждого числа (вещественные значения)
        scores = self.model.predict(X_new_scaled)[0]  # shape (total_numbers,)
        
        # Добавляем небольшой шум для разнообразия (уникальный seed)
        rng = random.Random(self.random_state + 777)
        scores = scores + [rng.gauss(0, 0.05) for _ in range(len(scores))]
        
        # Выбираем pick_count чисел с наибольшими scores
        top_indices = np.argsort(scores)[-self.pick_count:][::-1]
        selected = [i+1 for i in top_indices]
        return sorted(selected)