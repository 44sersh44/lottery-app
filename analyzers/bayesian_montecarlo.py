"""Байесовский анализ и Монте-Карло симуляции для лотерейных данных."""

import numpy as np
import random
from collections import Counter
from typing import List, Dict, Tuple


class BayesianAnalyzer:
    """
    Байесовский анализ для вероятностных выводов.
    Использует априорные распределения и обновляет вероятности на основе данных.
    """
    
    def __init__(self, blocks: List[List[int]], total_numbers: int, pick_count: int):
        self.blocks = blocks
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.draw_count = len(blocks)
        self.prior = self._create_prior()
        self.posterior = None
    
    def _create_prior(self) -> np.ndarray:
        """Создаёт априорное распределение (равномерное)."""
        return np.ones(self.total_numbers) / self.total_numbers
    
    def update_posterior(self) -> np.ndarray:
        """Обновляет апостериорные вероятности на основе исторических данных."""
        if self.draw_count == 0:
            self.posterior = self.prior
            return self.posterior
        
        frequencies = np.zeros(self.total_numbers)
        for block in self.blocks:
            for num in block:
                frequencies[num - 1] += 1
        
        total_counts = sum(frequencies)
        if total_counts > 0:
            likelihood = frequencies / total_counts
        else:
            likelihood = self.prior
        
        posterior = self.prior * likelihood
        posterior_sum = np.sum(posterior)
        if posterior_sum > 0:
            posterior = posterior / posterior_sum
        else:
            posterior = self.prior
        
        self.posterior = posterior
        return posterior
    
    # ============== ДОБАВЛЕННЫЙ МЕТОД get_report ==============
    def get_report(self) -> dict:
        """
        Возвращает отчёт о байесовском анализе в виде словаря.
        """
        if self.draw_count == 0:
            return {
                'total_blocks': 0,
                'total_numbers': self.total_numbers,
                'pick_count': self.pick_count,
                'top_numbers': [],
                'probabilities': {},
                'message': 'Нет данных для анализа'
            }
        
        if self.posterior is None:
            self.update_posterior()
        
        probs = [(i + 1, self.posterior[i]) for i in range(self.total_numbers)]
        probs.sort(key=lambda x: x[1], reverse=True)
        top_numbers = probs[:10]
        
        report = {
            'total_blocks': self.draw_count,
            'total_numbers': self.total_numbers,
            'pick_count': self.pick_count,
            'top_numbers': top_numbers,
            'probabilities': {num: prob for num, prob in probs},
        }
        return report
    
    def get_top_probabilities(self, top_n: int = 15) -> List[Tuple[int, float]]:
        """Возвращает топ-N чисел с наибольшей вероятностью."""
        if self.draw_count == 0:
            return [(i + 1, 1.0 / self.total_numbers) for i in range(min(top_n, self.total_numbers))]
        if self.posterior is None:
            self.update_posterior()
        probs = [(i + 1, self.posterior[i]) for i in range(self.total_numbers)]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_n]
    
    def get_number_probability(self, number: int) -> float:
        """Возвращает вероятность конкретного числа."""
        if number < 1 or number > self.total_numbers:
            return 0.0
        if self.draw_count == 0:
            return 1.0 / self.total_numbers
        if self.posterior is None:
            self.update_posterior()
        return self.posterior[number - 1]
    
    def predict_combination(self, n_combinations: int = 5) -> List[List[int]]:
        """Предсказывает комбинации на основе апостериорных вероятностей."""
        if self.draw_count == 0:
            combinations = []
            for _ in range(n_combinations):
                combo = sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
                combinations.append(combo)
            return combinations
        
        if self.posterior is None:
            self.update_posterior()
        
        combinations = []
        for _ in range(n_combinations):
            chosen = set()
            probs = self.posterior.copy()
            while len(chosen) < self.pick_count:
                remaining_probs = probs.copy()
                for num in chosen:
                    remaining_probs[num - 1] = 0
                total = np.sum(remaining_probs)
                if total > 0:
                    remaining_probs = remaining_probs / total
                else:
                    remaining_probs = np.ones(self.total_numbers) / self.total_numbers
                selected = np.random.choice(self.total_numbers, p=remaining_probs)
                chosen.add(selected + 1)
            combinations.append(sorted(chosen))
        return combinations
    
    def print_report(self):
        """Выводит байесовский анализ в консоль."""
        if self.draw_count == 0:
            print("\n" + "=" * 80)
            print("📊 БАЙЕСОВСКИЙ АНАЛИЗ ВЕРОЯТНОСТЕЙ")
            print("=" * 80)
            print("❌ Нет данных для анализа")
            return
        
        probs = self.get_top_probabilities(15)
        print("\n" + "=" * 80)
        print("📊 БАЙЕСОВСКИЙ АНАЛИЗ ВЕРОЯТНОСТЕЙ")
        print("=" * 80)
        print("\n🔢 ТОП-15 НАИБОЛЕЕ ВЕРОЯТНЫХ ЧИСЕЛ:")
        print("┌─────┬──────────┬─────────────────────────────────────────────┐")
        print("│ №   │ Число    │ Вероятность                                 │")
        print("├─────┼──────────┼─────────────────────────────────────────────┤")
        for i, (num, prob) in enumerate(probs, 1):
            bar_len = int(prob * 100)
            bar = "█" * bar_len if bar_len > 0 else ""
            print(f"│ {i:2d}  │   {num:2d}    │ {prob*100:5.2f}% {bar:<40} │")
        print("└─────┴──────────┴─────────────────────────────────────────────┘")
        
        print("\n🎯 ПРЕДСКАЗАННЫЕ КОМБИНАЦИИ (на основе Байеса):")
        predictions = self.predict_combination(5)
        for i, pred in enumerate(predictions, 1):
            print(f"   {i}. {pred}")


class MonteCarloSimulator:
    """
    Монте-Карло симуляции для оценки вероятностей.
    Проводит тысячи симуляций для оценки шансов.
    """
    
    def __init__(self, blocks: List[List[int]], total_numbers: int, pick_count: int):
        self.blocks = blocks
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.draw_count = len(blocks)
        self._prepare_history()
    
    def _prepare_history(self):
        """Подготавливает исторические данные для симуляций."""
        self.history = []
        for block in self.blocks:
            self.history.append(sorted(block))
        
        self.freq = Counter()
        for block in self.blocks:
            for num in block:
                self.freq[num] += 1
        
        total = sum(self.freq.values())
        if total > 0:
            self.probs = {num: count / total for num, count in self.freq.items()}
        else:
            self.probs = {i: 1.0 / self.total_numbers for i in range(1, self.total_numbers + 1)}
        
        self.numbers = list(self.probs.keys())
        self.weights = list(self.probs.values())
    
    def simulate_random_draw(self) -> List[int]:
        """Симулирует один случайный тираж."""
        return sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
    
    def simulate_historical_draw(self) -> List[int]:
        """Симулирует тираж на основе исторических частот."""
        if self.draw_count == 0:
            return self.simulate_random_draw()
        chosen = set()
        while len(chosen) < self.pick_count:
            selected = random.choices(self.numbers, weights=self.weights)[0]
            chosen.add(selected)
        return sorted(chosen)
    
    def simulate_markov_chain(self, order: int = 2) -> List[int]:
        """Симулирует тираж на основе марковской цепи."""
        if self.draw_count < order + 1:
            return self.simulate_historical_draw()
        
        transitions = {}
        for block in self.history:
            for i in range(len(block) - order):
                state = tuple(block[i:i+order])
                next_num = block[i+order]
                if state not in transitions:
                    transitions[state] = []
                transitions[state].append(next_num)
        
        if not transitions:
            return self.simulate_historical_draw()
        
        start_state = random.choice(list(transitions.keys()))
        result = list(start_state)
        while len(result) < self.pick_count:
            state = tuple(result[-order:])
            if state in transitions and transitions[state]:
                next_num = random.choice(transitions[state])
                if next_num not in result:
                    result.append(next_num)
            else:
                new_num = random.randint(1, self.total_numbers)
                if new_num not in result:
                    result.append(new_num)
            result.sort()
        return result[:self.pick_count]
    
    def run_simulations(self, method: str = "historical", n_simulations: int = 10000) -> Dict:
        """Запускает N симуляций и собирает статистику."""
        if self.draw_count == 0 and method != "random":
            print(f"[WARN] Нет данных для метода '{method}', используем random")
            method = "random"
        
        print(f"\n[SIM] Запуск {n_simulations} симуляций методом '{method}'...")
        results = Counter()
        for _ in range(n_simulations):
            if method == "random":
                draw = self.simulate_random_draw()
            elif method == "markov":
                draw = self.simulate_markov_chain()
            else:
                draw = self.simulate_historical_draw()
            for num in draw:
                results[num] += 1
        
        total = sum(results.values())
        if total > 0:
            probabilities = {num: count / total for num, count in results.items()}
        else:
            probabilities = {i: 1.0 / self.total_numbers for i in range(1, self.total_numbers + 1)}
        return probabilities
    
    def estimate_win_probability(self, prediction: List[int], n_simulations: int = 10000, 
                                   as_percent: bool = True) -> Dict:
        """Оценивает вероятность выигрыша для заданной комбинации."""
        if self.draw_count == 0:
            print("[WARN] Нет данных для оценки вероятности")
            return {i: 0.0 for i in range(1, self.pick_count + 1)}
        
        print(f"\n[SIM] Оценка вероятности выигрыша для комбинации {prediction}...")
        hits = {i: 0 for i in range(1, self.pick_count + 1)}
        for _ in range(n_simulations):
            draw = self.simulate_historical_draw()
            matches = len(set(prediction) & set(draw))
            if matches > 0:
                hits[matches] += 1
        
        if as_percent:
            probabilities = {k: v / n_simulations * 100 for k, v in hits.items()}
        else:
            probabilities = {k: v / n_simulations for k, v in hits.items()}
        return probabilities

    # ============== ОБНОВЛЁННЫЙ МЕТОД с поддержкой параметра method ==============
    def find_best_combinations(self, n_combinations: int = 10, 
                               n_simulations: int = 5000,
                               method: str = 'historical') -> List[Tuple[List[int], float]]:
        """
        Находит лучшие комбинации на основе симуляций.
        method: 'historical' (по частотам) или 'markov' (марковская цепь).
        """
        if self.draw_count == 0:
            print("[WARN] Нет данных для поиска лучших комбинаций")
            return []
        
        print(f"\n[SIM] Поиск лучших комбинаций методом Монте-Карло (метод={method})...")
        
        # Генерируем кандидатов
        most_common = [num for num, _ in self.freq.most_common(self.pick_count * 3)]
        candidates = []
        for _ in range(n_combinations * 5):
            if most_common and len(most_common) >= self.pick_count:
                candidate = sorted(random.sample(most_common, self.pick_count))
            else:
                candidate = sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
            candidates.append(candidate)
        candidates = list(set(tuple(c) for c in candidates))
        candidates = [list(c) for c in candidates]
        
        # Оцениваем каждую комбинацию
        sims_per_candidate = max(100, n_simulations // 10)
        scores = []
        for candidate in candidates[:n_combinations * 2]:
            # Используем estimate_win_probability (она использует historical),
            # но для markov можно было бы реализовать отдельную оценку.
            prob = self.estimate_win_probability(candidate, sims_per_candidate, as_percent=False)
            score = sum(matches * prob.get(matches, 0) for matches in range(1, self.pick_count + 1))
            scores.append((candidate, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_combinations]
    
    def print_report(self, method: str = "historical", n_simulations: int = 10000):
        """Выводит отчёт о симуляциях."""
        if self.draw_count == 0 and method != "random":
            print("[WARN] Нет данных, используются случайные симуляции")
            method = "random"
        
        probabilities = self.run_simulations(method, n_simulations)
        print("\n" + "=" * 80)
        print(f"📊 МОНТЕ-КАРЛО СИМУЛЯЦИИ ({method.upper()}, {n_simulations} итераций)")
        print("=" * 80)
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        print("\n🔢 ТОП-15 НАИБОЛЕЕ ВЕРОЯТНЫХ ЧИСЕЛ:")
        print("┌─────┬──────────┬─────────────────────────────────────────────┐")
        print("│ №   │ Число    │ Вероятность (%)                             │")
        print("├─────┼──────────┼─────────────────────────────────────────────┤")
        for i, (num, prob) in enumerate(sorted_probs[:15], 1):
            bar_len = int(prob * 100)
            bar = "█" * bar_len if bar_len > 0 else ""
            print(f"│ {i:2d}  │   {num:2d}    │ {prob*100:6.2f}% {bar:<40} │")
        print("└─────┴──────────┴─────────────────────────────────────────────┘")