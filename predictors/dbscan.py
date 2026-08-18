import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

# Проверяем доступность sklearn
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] sklearn не установлен, DBSCANPredictor будет использовать случайные прогнозы")


class DBSCANPredictor(BasePredictor):
    """Кластеризация DBSCAN предыдущих блоков, предсказание – центроид ближайшего кластера."""
    
    def __init__(self, total_numbers: int, pick_count: int, eps: float = 0.5, min_samples: int = 2, name: str = "DBSCAN"):
        # ✅ ИСПРАВЛЕНО: передаём total_numbers и pick_count в базовый класс
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None
        self.labels = None
        self.last_blocks = []
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not SKLEARN_OK or len(blocks) < 2:
            self.is_trained = True
            return

        # Признаки: суммы блоков
        X = np.array([sum(block) for block in blocks]).reshape(-1, 1)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.labels = db.labels_
        self.is_trained = True

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        if not SKLEARN_OK or self.labels is None or not self.last_blocks:
            return self._fallback()

        last_sum = sum(self.last_blocks[-1])
        # Ищем ближайший по сумме кластер (среднее арифметическое сумм блоков в кластере)
        cluster_means = {}
        for i, lbl in enumerate(self.labels):
            if lbl == -1:
                continue
            if lbl not in cluster_means:
                cluster_means[lbl] = []
            cluster_means[lbl].append(sum(self.last_blocks[i]))

        if not cluster_means:
            return self._fallback()

        best_cluster = min(cluster_means.keys(), key=lambda c: abs(np.mean(cluster_means[c]) - last_sum))
        target_sum = np.mean(cluster_means[best_cluster])
        return self._find_combo(target_sum)

    def predict(self, n_predictions: int = 1) -> List[List[int]]:
        """Возвращает несколько прогнозов."""
        return [self.predict_single() for _ in range(n_predictions)]

    def _find_combo(self, target_sum: float) -> List[int]:
        """Находит комбинацию чисел с суммой, близкой к target_sum."""
        for _ in range(2000):
            combo = sorted(np.random.choice(
                range(1, self.total_numbers + 1),
                self.pick_count,
                replace=False
            ).tolist())
            if abs(sum(combo) - target_sum) < 10:
                return combo
        # fallback
        return self._fallback()

    def _fallback(self) -> List[int]:
        """Случайный прогноз (если sklearn не доступен или нет данных)."""
        return sorted(np.random.choice(
            range(1, self.total_numbers + 1),
            self.pick_count,
            replace=False
        ).tolist())