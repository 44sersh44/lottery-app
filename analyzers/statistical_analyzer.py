"""Глубокий статистический анализ лотерейных данных."""

from collections import Counter
from typing import List, Dict, Tuple
import numpy as np


class StatisticalAnalyzer:
    """Анализирует статистику лотерейных данных."""
    
    def __init__(self, blocks: List[List[int]], total_numbers: int = 35, pick_count: int = 7):
        self.blocks = blocks
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.draw_count = len(blocks)
    
    def analyze_parity(self) -> Dict[str, float]:
        """Анализ чётных и нечётных чисел."""
        if self.draw_count == 0:
            return {"чётные": 0, "нечётные": 0, "чётные_среднее": 0, "нечётные_среднее": 0}
        
        even_count = 0
        odd_count = 0
        total = 0
        
        for block in self.blocks:
            for num in block:
                if num % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
                total += 1
        
        if total == 0:
            return {"чётные": 0, "нечётные": 0, "чётные_среднее": 0, "нечётные_среднее": 0}
        
        return {
            "чётные": round(even_count / total * 100, 1),
            "нечётные": round(odd_count / total * 100, 1),
            "чётные_среднее": round(even_count / self.draw_count, 2),
            "нечётные_среднее": round(odd_count / self.draw_count, 2)
        }
    
    def analyze_by_decade(self) -> Dict[str, Dict]:
        """Анализ распределения по десяткам."""
        if self.draw_count == 0:
            return {}
        
        decades = {
            "1-9": list(range(1, 10)),
            "10-19": list(range(10, 20)),
            "20-29": list(range(20, 30)),
            "30-39": list(range(30, min(40, self.total_numbers + 1))),
            "40-49": list(range(40, min(50, self.total_numbers + 1))),
            "50-54": list(range(50, self.total_numbers + 1))
        }
        
        # Убираем пустые десятки
        decades = {k: v for k, v in decades.items() if v}
        
        result = {}
        for decade_name, decade_numbers in decades.items():
            count = 0
            for block in self.blocks:
                for num in block:
                    if num in decade_numbers:
                        count += 1
            
            avg = count / self.draw_count
            result[decade_name] = {
                "частота": count,
                "среднее_на_тираж": round(avg, 2),
                "процент": round(avg / self.pick_count * 100, 1)
            }
        
        return result
    
    def analyze_sum(self) -> Dict[str, float]:
        """Анализ суммы чисел в тираже."""
        if self.draw_count == 0:
            return {"средняя": 0, "медиана": 0, "мин": 0, "макс": 0, "стандартное_отклонение": 0}
        
        sums = [sum(block) for block in self.blocks]
        
        return {
            "средняя": round(np.mean(sums), 1),
            "медиана": int(np.median(sums)),
            "мин": min(sums),
            "макс": max(sums),
            "стандартное_отклонение": round(np.std(sums), 1)
        }
    
    def analyze_pairs(self, top_n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """Анализ наиболее частых пар чисел."""
        if self.draw_count < 2:
            return []
        
        pairs = Counter()
        for block in self.blocks:
            sorted_block = sorted(block)
            for i in range(len(sorted_block)):
                for j in range(i + 1, len(sorted_block)):
                    pair = tuple(sorted([sorted_block[i], sorted_block[j]]))
                    pairs[pair] += 1
        
        return pairs.most_common(top_n)
    
    def analyze_triplets(self, top_n: int = 10) -> List[Tuple[Tuple[int, int, int], int]]:
        """Анализ наиболее частых троек чисел."""
        if self.draw_count < 3:
            return []
        
        triplets = Counter()
        for block in self.blocks:
            sorted_block = sorted(block)
            for i in range(len(sorted_block)):
                for j in range(i + 1, len(sorted_block)):
                    for k in range(j + 1, len(sorted_block)):
                        triplet = tuple(sorted([sorted_block[i], sorted_block[j], sorted_block[k]]))
                        triplets[triplet] += 1
        
        return triplets.most_common(top_n)
    
    def analyze_number_frequency(self, top_n: int = 15) -> List[Tuple[int, int]]:
        """Частотный анализ чисел."""
        if self.draw_count == 0:
            return []
        
        counter = Counter()
        for block in self.blocks:
            counter.update(block)
        return counter.most_common(top_n)
    
    def analyze_position_trends(self) -> Dict[int, Dict]:
        """Анализ по позициям (1-е число, 2-е и т.д.)."""
        if not self.blocks or self.draw_count == 0:
            return {}
        
        result = {}
        sorted_blocks = [sorted(block) for block in self.blocks]
        
        for pos in range(self.pick_count):
            values = [block[pos] for block in sorted_blocks]
            result[pos + 1] = {
                "среднее": round(np.mean(values), 1),
                "медиана": int(np.median(values)),
                "мин": min(values),
                "макс": max(values),
                "стандартное_отклонение": round(np.std(values), 1)
            }
        
        return result
    
    def analyze_gaps(self, top_n: int = 10) -> List[Tuple[int, Dict]]:
        """Анализ интервалов между выпадениями чисел."""
        if self.draw_count == 0:
            return []
        
        last_seen = {}
        gaps = {num: [] for num in range(1, self.total_numbers + 1)}
        
        for draw_index, block in enumerate(self.blocks):
            for num in block:
                if num in last_seen:
                    gap = draw_index - last_seen[num]
                    gaps[num].append(gap)
                last_seen[num] = draw_index
        
        result = []
        for num in range(1, self.total_numbers + 1):
            if gaps[num]:
                avg_gap = round(np.mean(gaps[num]), 1)
                max_gap = max(gaps[num])
                current_gap = self.draw_count - last_seen.get(num, 0)
            else:
                avg_gap = 0
                max_gap = 0
                current_gap = self.draw_count - last_seen.get(num, 0)
            
            result.append((num, {
                "средний": avg_gap,
                "максимальный": max_gap,
                "текущий": current_gap
            }))
        
        result.sort(key=lambda x: x[1]['текущий'], reverse=True)
        return result[:top_n]
    
    def analyze_consecutive(self) -> Dict[str, float]:
        """Анализ последовательных чисел (идущих подряд)."""
        if self.draw_count == 0:
            return {"тиражей_с_последовательными": 0, "процент": 0, "максимум_подряд": 0}
        
        consecutive_count = 0
        max_consecutive = 0
        
        for block in self.blocks:
            sorted_block = sorted(block)
            current_streak = 1
            has_consecutive = False
            
            for i in range(1, len(sorted_block)):
                if sorted_block[i] == sorted_block[i-1] + 1:
                    current_streak += 1
                    has_consecutive = True
                else:
                    if current_streak > max_consecutive:
                        max_consecutive = current_streak
                    current_streak = 1
            
            if current_streak > max_consecutive:
                max_consecutive = current_streak
            
            if has_consecutive:
                consecutive_count += 1
        
        return {
            "тиражей_с_последовательными": consecutive_count,
            "процент": round(consecutive_count / self.draw_count * 100, 1),
            "максимум_подряд": max_consecutive
        }
    
    def analyze_by_draw_position(self) -> Dict[int, Dict]:
        """Анализ распределения чисел по позициям в тираже."""
        if not self.blocks or self.draw_count == 0:
            return {}
        
        result = {}
        sorted_blocks = [sorted(block) for block in self.blocks]
        
        for pos in range(self.pick_count):
            position_counts = Counter()
            for block in sorted_blocks:
                position_counts[block[pos]] += 1
            
            if position_counts:
                most_common = position_counts.most_common(1)[0]
                result[pos + 1] = {
                    "самое_частое": most_common[0],
                    "частота": most_common[1],
                    "топ_3": position_counts.most_common(3)
                }
            else:
                result[pos + 1] = {
                    "самое_частое": 0,
                    "частота": 0,
                    "топ_3": []
                }
        
        return result
    
    # ============== ДОБАВЛЕННЫЙ МЕТОД ==============
    def get_full_report(self) -> dict:
        """
        Возвращает полный статистический отчёт в виде словаря.
        Используется для вывода в консоль без печати внутри класса.
        """
        if self.draw_count == 0:
            return {
                'total_blocks': 0,
                'total_numbers': self.total_numbers,
                'pick_count': self.pick_count,
                'error': 'Нет данных'
            }
        
        return {
            'total_blocks': self.draw_count,
            'total_numbers': self.total_numbers,
            'pick_count': self.pick_count,
            'frequencies': self.analyze_number_frequency(self.total_numbers),
            'top_frequencies': self.analyze_number_frequency(15),
            'parity_stats': self.analyze_parity(),
            'decade_stats': self.analyze_by_decade(),
            'pairs': self.analyze_pairs(5),
            'triples': self.analyze_triplets(5),
            'consecutive': self.analyze_consecutive(),
            'sum_stats': self.analyze_sum(),
            'position_trends': self.analyze_position_trends(),
            'gaps': self.analyze_gaps(10),
            'position_analysis': self.analyze_by_draw_position()
        }
    
    # ============== ОРИГИНАЛЬНЫЙ МЕТОД PRINT_FULL_REPORT (оставляем) ==============
    def print_full_report(self):
        """Выводит полный отчёт по всем анализам."""
        if self.draw_count == 0:
            print("\n" + "=" * 80)
            print("📊 ГЛУБОКИЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
            print("=" * 80)
            print("❌ Нет данных для анализа")
            print("=" * 80)
            return
        
        print("\n" + "=" * 80)
        print("📊 ГЛУБОКИЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        print("=" * 80)
        print(f"📈 Всего тиражей: {self.draw_count}")
        print(f"🎯 Диапазон чисел: 1-{self.total_numbers}")
        print(f"📦 Чисел в тираже: {self.pick_count}")
        
        # 1. Чётность
        parity = self.analyze_parity()
        print("\n" + "=" * 80)
        print("🔢 1. АНАЛИЗ ЧЁТНОСТИ:")
        print(f"   Чётные:   {parity['чётные']}% (в среднем {parity['чётные_среднее']} чисел из {self.pick_count})")
        print(f"   Нечётные: {parity['нечётные']}% (в среднем {parity['нечётные_среднее']} чисел из {self.pick_count})")
        
        # 2. Десятки
        decades = self.analyze_by_decade()
        print("\n" + "=" * 80)
        print("📊 2. РАСПРЕДЕЛЕНИЕ ПО ДЕСЯТКАМ:")
        for decade, data in decades.items():
            bar = "█" * int(data['процент'] / 2)
            print(f"   {decade:6}: {data['процент']:5.1f}% {bar} ({data['среднее_на_тираж']} чисел/тираж)")
        
        # 3. Суммы
        sums = self.analyze_sum()
        print("\n" + "=" * 80)
        print("💰 3. АНАЛИЗ СУММЫ ЧИСЕЛ В ТИРАЖЕ:")
        print(f"   Средняя сумма:     {sums['средняя']}")
        print(f"   Медиана:           {sums['медиана']}")
        print(f"   Диапазон:          {sums['мин']} - {sums['макс']}")
        print(f"   Стандартное отклонение: ±{sums['стандартное_отклонение']}")
        
        # 4. Частые пары
        print("\n" + "=" * 80)
        print("🤝 4. НАИБОЛЕЕ ЧАСТЫЕ ПАРЫ (топ-10):")
        pairs = self.analyze_pairs(10)
        for i, (pair, count) in enumerate(pairs, 1):
            prob = round(count / self.draw_count * 100, 1)
            bar = "█" * min(20, count)
            print(f"   {i:2d}. {pair[0]:2d} и {pair[1]:2d} → {count} раз(а) ({prob}%) {bar}")
        
        # 5. Частые тройки
        print("\n" + "=" * 80)
        print("👥 5. НАИБОЛЕЕ ЧАСТЫЕ ТРОЙКИ (топ-10):")
        triplets = self.analyze_triplets(10)
        for i, (triplet, count) in enumerate(triplets, 1):
            prob = round(count / self.draw_count * 100, 1)
            print(f"   {i:2d}. {triplet[0]:2d},{triplet[1]:2d},{triplet[2]:2d} → {count} раз(а) ({prob}%)")
        
        # 6. Топ частотности чисел
        print("\n" + "=" * 80)
        print("🔥 6. САМЫЕ ЧАСТЫЕ ЧИСЛА (топ-15):")
        freq = self.analyze_number_frequency(15)
        max_freq = max(f[1] for f in freq) if freq else 1
        for i, (num, count) in enumerate(freq, 1):
            bar = "█" * int(count / max_freq * 30)
            prob = round(count / self.draw_count * 100, 1)
            print(f"   {i:2d}. {num:3d} → {count:3d} раз(а) ({prob:4.1f}%) {bar}")
        
        # 7. Самые редкие числа
        print("\n" + "=" * 80)
        print("🧊 7. САМЫЕ РЕДКИЕ ЧИСЛА (топ-10):")
        all_nums = set(range(1, self.total_numbers + 1))
        freq_dict = dict(self.analyze_number_frequency(self.total_numbers))
        rare = [(num, freq_dict.get(num, 0)) for num in all_nums]
        rare.sort(key=lambda x: x[1])
        for i, (num, count) in enumerate(rare[:10], 1):
            print(f"   {i:2d}. {num:3d} → {count} раз(а)")
        
        # 8. Последовательные числа
        print("\n" + "=" * 80)
        print("📏 8. АНАЛИЗ ПОСЛЕДОВАТЕЛЬНЫХ ЧИСЕЛ (идущих подряд):")
        consecutive = self.analyze_consecutive()
        print(f"   Тиражи с последовательными числами: {consecutive['тиражей_с_последовательными']} ({consecutive['процент']}%)")
        print(f"   Максимум подряд идущих чисел в одном тираже: {consecutive['максимум_подряд']}")
        
        # 9. Анализ по позициям
        print("\n" + "=" * 80)
        print("📍 9. АНАЛИЗ ПО ПОЗИЦИЯМ (1-е, 2-е... число в тираже):")
        positions = self.analyze_position_trends()
        for pos, data in positions.items():
            print(f"   Позиция {pos}: среднее={data['среднее']}, медиана={data['медиана']}, мин={data['мин']}, макс={data['макс']}")
        
        # 10. Какие числа давно не выпадали
        print("\n" + "=" * 80)
        print("⏰ 10. ЧИСЛА, КОТОРЫЕ ДАВНО НЕ ВЫПАДАЛИ (топ-10):")
        gaps = self.analyze_gaps(10)
        for i, (num, gap_data) in enumerate(gaps, 1):
            print(f"   {i:2d}. Число {num:2d} → не выпадало {gap_data['текущий']} тиражей (макс пропуск: {gap_data['максимальный']})")
        
        # 11. Какое число чаще всего на каждой позиции
        print("\n" + "=" * 80)
        print("🎯 11. САМОЕ ЧАСТОЕ ЧИСЛО НА КАЖДОЙ ПОЗИЦИИ:")
        pos_analysis = self.analyze_by_draw_position()
        for pos, data in pos_analysis.items():
            print(f"   Позиция {pos}: чаще всего {data['самое_частое']} (выпадало {data['частота']} раз)")
        
        print("\n" + "=" * 80)
        print("✅ Анализ завершён")
        print("=" * 80)