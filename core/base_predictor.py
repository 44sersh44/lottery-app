"""Абстрактный базовый класс для всех предикторов."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple, Optional


class BasePredictor(ABC):
    """Единый интерфейс для методов прогнозирования."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.is_trained = False
        
        # Флаги для двойной лотереи
        self.is_double: bool = False
        self.total_numbers: int = 0
        self.pick_count: int = 0
        self.total_numbers2: int = 0
        self.pick_count2: int = 0
        self.blocks2: List[List[int]] = []

    @abstractmethod
    def fit(self, blocks: List[List[int]]) -> None:
        """
        Обучить модель на исторических блоках (порядок от старых к новым).
        
        Для двойной лотереи этот метод обучает только ПЕРВОЕ поле.
        Второе поле должно обучаться отдельно или через дополнительный параметр.
        """
        pass

    @abstractmethod
    def predict(self, n_predictions: int = 1) -> Union[List[List[int]], List[Tuple[List[int], List[int]]]]:
        """
        Вернуть топ-N предсказанных блоков.
        
        Returns:
            - Для обычной лотереи: список блоков List[List[int]]
            - Для двойной лотереи: список кортежей List[Tuple[pole1, pole2]]
        """
        pass

    def predict_single(self) -> Union[List[int], Tuple[List[int], List[int]]]:
        """
        Вернуть один прогноз.
        
        Returns:
            - Для обычной лотереи: список чисел List[int]
            - Для двойной лотереи: кортеж (поле1, поле2)
        """
        results = self.predict(1)
        return results[0] if results else ([] if not self.is_double else ([], []))

    def supports_double(self) -> bool:
        """
        Поддерживает ли модель двойную лотерею.
        
        По умолчанию False. Модели, которые поддерживают, должны переопределить.
        """
        return False

    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        """
        Обучить модель на данных двойной лотереи (оба поля).
        
        Базовая реализация вызывает fit() для первого поля,
        а второе поле сохраняет, но не обучает.
        """
        self.is_double = True
        self.blocks2 = blocks2
        # По умолчанию просто сохраняем длины
        if blocks2 and len(blocks2) > 0:
            self.pick_count2 = len(blocks2[0]) if blocks2[0] else 0
        self.fit(blocks1)

    def set_lottery_params(self, total_numbers: int, pick_count: int, 
                           total_numbers2: int = 0, pick_count2: int = 0) -> None:
        """
        Установить параметры лотереи.
        
        Args:
            total_numbers: Максимальное число в первом поле
            pick_count: Количество чисел в первом поле
            total_numbers2: Максимальное число во втором поле (для double)
            pick_count2: Количество чисел во втором поле (для double)
        """
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        
        if total_numbers2 > 0 or pick_count2 > 0:
            self.is_double = True
            self.total_numbers2 = total_numbers2
            self.pick_count2 = pick_count2

    def _fallback_single(self, n_predictions: int = 1) -> List[List[int]]:
        """
        Заглушка для обычной лотереи (случайные числа).
        """
        import random
        
        results = []
        for _ in range(n_predictions):
            block = sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
            results.append(block)
        
        return results

    def _fallback_double(self, n_predictions: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Заглушка для двойной лотереи (случайные числа).
        """
        import random
        
        results = []
        for _ in range(n_predictions):
            # Первое поле
            p1 = sorted(random.sample(range(1, self.total_numbers + 1), self.pick_count))
            # Второе поле
            if self.total_numbers2 > 0:
                p2 = [random.randint(1, self.total_numbers2)]
            else:
                p2 = []
            results.append((p1, p2))
        
        return results

    def __repr__(self) -> str:
        double_marker = " [DOUBLE]" if self.is_double else ""
        return f"{self.name} (trained={self.is_trained}{double_marker})"


class DoubleLotteryPredictor(BasePredictor):
    """
    Базовый класс для предикторов, специально созданных для двойной лотереи.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.is_double = True
        self.predictor1: Optional[BasePredictor] = None
        self.predictor2: Optional[BasePredictor] = None

    @abstractmethod
    def fit_double(self, blocks1: List[List[int]], blocks2: List[List[int]]) -> None:
        """
        Обучить модель на данных двойной лотереи (оба поля).
        """
        pass

    def fit(self, blocks: List[List[int]]) -> None:
        """
        Для совместимости с BasePredictor.
        Для двойной лотереи используйте fit_double().
        """
        raise NotImplementedError("Для двойной лотереи используйте fit_double()")

    def predict(self, n_predictions: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Вернуть топ-N предсказанных блоков для двойной лотереи.
        """
        if not self.is_trained:
            return self._fallback_double(n_predictions)
        
        results = self._predict_double(n_predictions)
        return results

    @abstractmethod
    def _predict_double(self, n_predictions: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Внутренний метод предсказания для двойной лотереи.
        """
        pass

    def supports_double(self) -> bool:
        """Поддерживает двойную лотерею."""
        return True