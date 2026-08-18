import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from .base import BasePredictor

# Проверяем доступность sklearn
try:
    from sklearn.naive_bayes import GaussianNB
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] sklearn не установлен, GaussianNBPredictor будет использовать случайные прогнозы")

# Проверяем доступность pandas (для qcut)
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    print("[WARN] pandas не установлен, GaussianNBPredictor будет использовать numpy.digitize")


class GaussianNBPredictor(BasePredictor):
    """Наивный байес (гауссов) – предсказываем сумму следующего блока."""
    
    def __init__(self, total_numbers: int, pick_count: int, name: str = "GaussianNB"):
        # ✅ Исправлено: передаём total_numbers и pick_count
        super().__init__(total_numbers, pick_count)
        self.name = name
        self.model = None
        self.last_blocks = []
        self.class_means = {}
        self.is_trained = False

    def fit(self, blocks: List[List[int]]) -> None:
        self.last_blocks = blocks
        if not SKLEARN_OK or len(blocks) < 2:
            self.is_trained = True
            return

        X = np.array([sum(block) for block in blocks[:-1]]).reshape(-1, 1)
        y = np.array([sum(block) for block in blocks[1:]])

        # Дискретизируем y в 5 классов (квантили)
        if PANDAS_OK:
            y_discrete = pd.qcut(y, q=5, labels=False)
        else:
            # fallback без pandas
            bins = np.percentile(y, np.linspace(0, 100, 6))[1:-1]
            y_discrete = np.digitize(y, bins=bins)

        self.model = GaussianNB()
        self.model.fit(X, y_discrete)

        # Вычисляем среднюю сумму для каждого класса
        unique_classes = np.unique(y_discrete)
        for cls in unique_classes:
            indices = np.where(y_discrete == cls)[0]
            self.class_means[cls] = np.mean(y[indices])

        self.is_trained = True

    def predict_single(self) -> List[int]:
        """Возвращает один прогноз."""
        if not SKLEARN_OK or self.model is None or not self.last_blocks:
            return self._fallback()

        last_sum = sum(self.last_blocks[-1])
        pred_class = self.model.predict([[last_sum]])[0]
        target_sum = self.class_means.get(pred_class, last_sum)
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
        return self._fallback()

    def _fallback(self) -> List[int]:
        """Случайный прогноз (если sklearn не доступен или нет данных)."""
        return sorted(np.random.choice(
            range(1, self.total_numbers + 1),
            self.pick_count,
            replace=False
        ).tolist())