# services/stats_service.py
import os
from collections import Counter
from typing import List, Optional, Dict, Any
from services.data_service import DataService
from analyzers.statistical_analyzer import StatisticalAnalyzer
from visualization.charts import LotteryCharts
from analyzers.statistical_analyzer import StatisticalAnalyzer

class StatsService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def show_frequency(self, top: int = 10) -> List[tuple]:
        data = self.data_service.current_data
        if not data or data.blocks_count() == 0:
            return []
        counter = Counter()
        for block in data.blocks:
            counter.update(block)
        return counter.most_common(top)

    def show_last_blocks(self, n: int = 10) -> List[dict]:
        data = self.data_service.current_data
        if not data:
            return []
        result = []
        for i in range(min(n, data.blocks_count())):
            block = data.blocks[i]
            meta = data.metadata[i] if i < len(data.metadata) else {}
            row = {
                'index': i+1,
                'block': block,
                'block2': data.blocks2[i] if data.is_double() and i < len(data.blocks2) else None,
                'draw_id': meta.get('draw_id', ''),
                'date': meta.get('date', ''),
                'time': meta.get('time', '')
            }
            result.append(row)
        return result

    def clear_history(self) -> bool:
        profile = self.data_service.profile_manager.current_profile
        if profile:
            profile.prediction_history = []
            profile.save()
            return True
        return False

    def show_statistical_analysis(self, field: int) -> dict:
        data = self.data_service.current_data
        if not data:
            return {}
        if field == 1:
            blocks = data.blocks
            total = data.total_numbers
            pick = data.pick_count
        else:
            blocks = data.blocks2
            total = data.total_numbers2
            pick = data.pick_count2
        analyzer = StatisticalAnalyzer(blocks, total, pick)
        # предполагаем, что есть метод get_full_report()
        report = analyzer.get_full_report()  # если нет, можно вызвать print_report и перехватить вывод, но лучше реализовать
        return report

    def show_charts(self, field: int, chart_type: int, predictions: Optional[List[int]] = None):
        data = self.data_service.current_data
        if not data:
            return
        if chart_type == 1:
            LotteryCharts.plot_frequency(data, field=field)
        elif chart_type == 2:
            LotteryCharts.plot_trend(data, field=field)
        elif chart_type == 3:
            LotteryCharts.plot_heatmap(data, field=field)
        elif chart_type == 4:
            LotteryCharts.plot_position_trend(data, field=field)
        elif chart_type == 5 and predictions:
            LotteryCharts.plot_comparison(data, predictions, field=field)