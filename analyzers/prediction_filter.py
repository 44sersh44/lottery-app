"""Фильтрация прогнозов на основе статистического анализа."""

from typing import List, Tuple, Dict
import numpy as np
from analyzers.statistical_analyzer import StatisticalAnalyzer


class PredictionFilter:
    """Фильтрует прогнозы по статистическим правилам."""
    
    def __init__(self, analyzer: StatisticalAnalyzer):
        self.analyzer = analyzer
        self.stats = self._collect_stats()
        self.decades = self._get_decades()
    
    def _collect_stats(self) -> Dict:
        """Собирает статистику для фильтрации."""
        parity = self.analyzer.analyze_parity()
        decades = self.analyzer.analyze_by_decade()
        sums = self.analyzer.analyze_sum()
        
        return {
            "even_ratio": parity['чётные'] / 100,
            "odd_ratio": parity['нечётные'] / 100,
            "decades": decades,
            "avg_sum": sums['средняя'],
            "std_sum": sums['стандартное_отклонение']
        }
    
    def _get_decades(self) -> List[str]:
        """Динамически определяет десятки на основе total_numbers."""
        total = self.analyzer.total_numbers
        decades = []
        start = 1
        while start <= total:
            end = min(start + 9, total)
            decades.append(f"{start}-{end}")
            start += 10
        return decades

    def _convert_to_python_int(self, obj):
        """Преобразует numpy типы в обычные int."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, list):
            return [self._convert_to_python_int(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._convert_to_python_int(x) for x in obj)
        return obj

    def filter_prediction(self, prediction: List[int], pick_count: int = 7) -> Tuple[bool, List[str]]:
        """
        Проверяет, соответствует ли прогноз статистике.
        
        Args:
            prediction: Список чисел для проверки
            pick_count: Ожидаемое количество чисел (7 для поля1, 1 для поля2)
        
        Returns:
            (соответствует, список нарушений)
        """
        # Проверка на пустой прогноз
        if not prediction:
            return False, ["пустой прогноз"]
        
        # Для поля2 (1 число) — упрощённая проверка
        if pick_count == 1:
            num = prediction[0] if prediction else None
            if num is None:
                return False, ["нет числа"]
            if num < 1 or num > self.analyzer.total_numbers:
                return False, [f"число вне диапазона: {num}"]
            return True, []
        
        violations = []
        
        # 1. Проверка чётности (допуск ±1 от среднего)
        even_count = sum(1 for x in prediction if x % 2 == 0)
        expected_even = int(len(prediction) * self.stats['even_ratio'])
        if not (expected_even - 1 <= even_count <= expected_even + 1):
            violations.append(f"чётность: {even_count} (ожидается ~{expected_even})")
        
        # 2. Проверка распределения по десяткам (динамические десятки)
        decade_counts = [0] * len(self.decades)
        for num in prediction:
            for idx, decade in enumerate(self.decades):
                start, end = map(int, decade.split('-'))
                if start <= num <= end:
                    decade_counts[idx] += 1
                    break
        
        for i, decade in enumerate(self.decades):
            if decade in self.stats['decades']:
                expected = self.stats['decades'][decade]['среднее_на_тираж']
                if not (max(0, int(expected) - 1) <= decade_counts[i] <= int(expected) + 2):
                    violations.append(f"десяток {decade}: {decade_counts[i]} (ожидается ~{expected:.1f})")
        
        # 3. Проверка суммы
        total = sum(prediction)
        min_sum = self.stats['avg_sum'] - self.stats['std_sum'] * 1.5
        max_sum = self.stats['avg_sum'] + self.stats['std_sum'] * 1.5
        if not (min_sum <= total <= max_sum):
            violations.append(f"сумма: {total} (ожидается {min_sum:.0f}-{max_sum:.0f})")
        
        # 4. Проверка на слишком много чисел из одного десятка
        if max(decade_counts) > 3:
            violations.append(f"слишком много чисел из одного десятка: {max(decade_counts)}")
        
        # 5. Проверка на слишком много чётных/нечётных
        if even_count < 2 or even_count > 5:
            violations.append(f"слишком {'мало' if even_count < 2 else 'много'} чётных: {even_count}")
        
        # 6. Проверка на подряд идущие числа (более 2 подряд)
        sorted_pred = sorted(prediction)
        consecutive = 1
        for i in range(1, len(sorted_pred)):
            if sorted_pred[i] == sorted_pred[i-1] + 1:
                consecutive += 1
            else:
                if consecutive > 2:
                    violations.append(f"подряд идущие числа: {consecutive} (max 2)")
                consecutive = 1
        if consecutive > 2:
            violations.append(f"подряд идущие числа: {consecutive} (max 2)")
        
        return len(violations) == 0, violations
    
    def get_best_predictions(self, predictions: List[Tuple[List[int], List[int], float]], 
                             max_results: int = 5, pick_count: int = 7) -> List[Tuple[List[int], List[int], float]]:
        """
        Возвращает только лучшие прогнозы, отфильтрованные по статистике.
        
        Args:
            predictions: Список прогнозов (поле1, поле2, уверенность)
            max_results: Максимальное количество результатов
            pick_count: Ожидаемое количество чисел (7 для поля1, 1 для поля2)
        """
        if not predictions:
            return []
        
        filtered = []
        rejected = 0
        
        for p1, p2, conf in predictions:
            # Преобразуем numpy типы
            p1 = self._convert_to_python_int(p1)
            p2 = self._convert_to_python_int(p2)
            
            is_ok, violations = self.filter_prediction(p1, pick_count)
            if is_ok:
                filtered.append((p1, p2, conf))
            else:
                rejected += 1
                if len(violations) > 0:
                    # Показываем только первое нарушение для краткости
                    print(f"  ⚠️ Отброшен: {p1} → {violations[0]}")
        
        print(f"\n[STATS] Принято: {len(filtered)}, Отброшено: {rejected}")
        
        if not filtered:
            print("\n⚠️ Все прогнозы отброшены! Показываю первые 3 без фильтрации.")
            return predictions[:min(max_results, len(predictions))]
        
        # Сортируем по уверенности
        filtered.sort(key=lambda x: x[2], reverse=True)
        return filtered[:max_results]