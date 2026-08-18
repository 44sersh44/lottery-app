"""
Единое хранилище лотерейных данных с поддержкой обычных и двойных лотерей.
Порядок блоков: blocks[0] — САМЫЙ СВЕЖИЙ (последний по времени),
              blocks[-1] — САМЫЙ СТАРЫЙ (первый по времени).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from collections import Counter
import copy


class LotteryData:
    """
    Контейнер для данных лотереи.
    Для обычной лотереи используется только blocks.
    Для двойной — blocks1 и blocks2.
    """
    
    def __init__(self, lottery_type: str = "single"):
        """
        Args:
            lottery_type: "single" или "double"
        """
        self.lottery_type = lottery_type
        self.blocks: List[List[int]] = []          # для single или поле1 double
        self.blocks2: List[List[int]] = []         # только для double (поле2)
        self.metadata: List[Dict[str, Any]] = []   # метаданные для каждого блока (дата, время, тираж)
        
        # Параметры лотереи (заполняются из профиля)
        self.pick_count: int = 0
        self.total_numbers: int = 0
        self.pick_count2: int = 0
        self.total_numbers2: int = 0
    
    def is_double(self) -> bool:
        return self.lottery_type == "double"
    
    def blocks_count(self) -> int:
        """Количество блоков (для double — по полю 1, оно же ведущее)"""
        return len(self.blocks)
    
    def _validate_block(self, block: List[int], total_numbers: int, pick_count: int, field_name: str = "поле") -> None:
        """
        Проверяет корректность блока чисел.
        
        Args:
            block: Список чисел для проверки
            total_numbers: Максимальное допустимое число
            pick_count: Ожидаемое количество чисел
            field_name: Название поля для сообщения об ошибке
        
        Raises:
            ValueError: Если блок некорректен
        """
        if not isinstance(block, list):
            raise ValueError(f"{field_name} должно быть списком, получено {type(block)}")
        
        if len(block) != pick_count:
            raise ValueError(f"{field_name}: должно быть {pick_count} чисел, получено {len(block)}")
        
        for num in block:
            if not isinstance(num, (int, float)) or int(num) != num:
                raise ValueError(f"{field_name}: число {num} должно быть целым")
            num = int(num)
            if not 1 <= num <= total_numbers:
                raise ValueError(f"{field_name}: число {num} вне диапазона 1-{total_numbers}")
        
        if len(set(block)) != len(block):
            raise ValueError(f"{field_name}: обнаружены повторяющиеся числа: {block}")
    
    def fix_sync(self) -> None:
        """
        Восстанавливает синхронизацию blocks, blocks2 и metadata.
        Выравнивает все списки по минимальной длине.
        """
        if self.is_double():
            min_len = min(len(self.blocks), len(self.blocks2))
            
            if len(self.blocks) != min_len:
                removed = len(self.blocks) - min_len
                self.blocks = self.blocks[:min_len]
                if removed > 0:
                    print(f"[SYNC] Удалено {removed} блоков из поле1 (несинхронизированных)")
            
            if len(self.blocks2) != min_len:
                removed = len(self.blocks2) - min_len
                self.blocks2 = self.blocks2[:min_len]
                if removed > 0:
                    print(f"[SYNC] Удалено {removed} блоков из поле2 (несинхронизированных)")
            
            if len(self.metadata) != min_len:
                self.metadata = self.metadata[:min_len]
        else:
            # Для обычной лотереи синхронизируем только blocks и metadata
            if len(self.metadata) != len(self.blocks):
                self.metadata = self.metadata[:len(self.blocks)]
    
    def get_latest_block(self) -> Optional[Union[List[int], Tuple[List[int], List[int]]]]:
        """
        Возвращает самый свежий блок (блоки).
        Для single: список чисел.
        Для double: кортеж (поле1, поле2).
        """
        if not self.blocks:
            return None
        
        # Автоматическая синхронизация при получении
        if self.is_double():
            self.fix_sync()
            return (self.blocks[0], self.blocks2[0] if self.blocks2 else [])
        return self.blocks[0]
    
    def get_oldest_block(self) -> Optional[Union[List[int], Tuple[List[int], List[int]]]]:
        """Самый старый блок."""
        if not self.blocks:
            return None
        
        if self.is_double():
            self.fix_sync()
            return (self.blocks[-1], self.blocks2[-1] if self.blocks2 else [])
        return self.blocks[-1]
    
    def get_recent_blocks(self, n: int = 10, field: int = 1) -> List[List[int]]:
        """Возвращает последние N блоков (свежие)."""
        if field == 1:
            blocks = self.blocks
        elif field == 2 and self.is_double():
            blocks = self.blocks2
        else:
            raise ValueError(f"Неверное поле {field}")
        return blocks[:min(n, len(blocks))]
    
    def add_block(self, block: Union[List[int], Tuple[List[int], List[int]]],
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавить блок в НАЧАЛО (как самый свежий).
        Для single: block = список чисел.
        Для double: block = (поле1, поле2).
        """
        if self.is_double():
            if not isinstance(block, (tuple, list)) or len(block) != 2:
                raise ValueError("Для двойной лотереи block должен быть (поле1, поле2)")
            
            block1, block2 = block
            self._validate_block(list(block1), self.total_numbers, self.pick_count, "Поле1")
            self._validate_block(list(block2), self.total_numbers2, self.pick_count2, "Поле2")
            
            self.blocks.insert(0, list(block1))
            self.blocks2.insert(0, list(block2))
        else:
            self._validate_block(list(block), self.total_numbers, self.pick_count, "Поле")
            self.blocks.insert(0, list(block))
        
        if metadata:
            self.metadata.insert(0, metadata)
        else:
            self.metadata.insert(0, {})
        
        # Проверка синхронизации после добавления
        if self.is_double():
            assert len(self.blocks) == len(self.blocks2), "Ошибка синхронизации после добавления блока!"
    
    def append_block(self, block: Union[List[int], Tuple[List[int], List[int]]],
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Добавить блок в КОНЕЦ (как самый старый)."""
        if self.is_double():
            if not isinstance(block, (tuple, list)) or len(block) != 2:
                raise ValueError("Для двойной лотереи block должен быть (поле1, поле2)")
            
            block1, block2 = block
            self._validate_block(list(block1), self.total_numbers, self.pick_count, "Поле1")
            self._validate_block(list(block2), self.total_numbers2, self.pick_count2, "Поле2")
            
            self.blocks.append(list(block1))
            self.blocks2.append(list(block2))
        else:
            self._validate_block(list(block), self.total_numbers, self.pick_count, "Поле")
            self.blocks.append(list(block))
        
        if metadata:
            self.metadata.append(metadata)
        else:
            self.metadata.append({})
        
        if self.is_double():
            assert len(self.blocks) == len(self.blocks2), "Ошибка синхронизации после добавления блока!"
    
    def get_blocks_for_training(self, field: int = 1) -> List[List[int]]:
        """
        Возвращает блоки для обучения (от старых к новым).
        
        Args:
            field: 1 — первое поле, 2 — второе поле
        
        Returns:
            Список блоков в порядке от старых к новым (для обучения)
        """
        if field == 1:
            return self.blocks[::-1]  # быстрее, чем list(reversed(self.blocks))
        elif field == 2:
            if not self.is_double():
                raise ValueError("Второе поле доступно только для двойной лотереи")
            return self.blocks2[::-1]
        else:
            raise ValueError(f"Неверное поле {field}. Допустимые значения: 1 или 2")
    
    def split_by_time(self, test_ratio: float = 0.2) -> Tuple['LotteryData', 'LotteryData']:
        """
        Разделяет данные на обучающую (старые блоки) и тестовую (свежие).
        Возвращает (train_data, test_data).
        Порядок: train — старые (конец списка), test — свежие (начало списка).
        
        Args:
            test_ratio: Доля свежих данных для теста (0.0-1.0)
        
        Raises:
            ValueError: Если недостаточно данных или test_ratio некорректен
        """
        n = self.blocks_count()
        
        if n < 2:
            raise ValueError(f"Недостаточно данных для разделения: нужно минимум 2 блока, есть {n}")
        
        if not 0 < test_ratio < 1:
            raise ValueError(f"test_ratio должен быть между 0 и 1, получен {test_ratio}")
        
        # Минимум 1 блок в тестовую и 1 в обучающую
        split_idx = max(1, min(n - 1, int(n * (1 - test_ratio))))
        
        train = LotteryData(self.lottery_type)
        test = LotteryData(self.lottery_type)
        
        # Копируем параметры
        train.pick_count = self.pick_count
        train.total_numbers = self.total_numbers
        train.pick_count2 = self.pick_count2
        train.total_numbers2 = self.total_numbers2
        test.pick_count = self.pick_count
        test.total_numbers = self.total_numbers
        test.pick_count2 = self.pick_count2
        test.total_numbers2 = self.total_numbers2
        
        if self.is_double():
            self.fix_sync()  # Синхронизация перед разделением
            train.blocks = self.blocks[split_idx:]
            train.blocks2 = self.blocks2[split_idx:]
            test.blocks = self.blocks[:split_idx]
            test.blocks2 = self.blocks2[:split_idx]
        else:
            train.blocks = self.blocks[split_idx:]
            test.blocks = self.blocks[:split_idx]
        
        train.metadata = self.metadata[split_idx:]
        test.metadata = self.metadata[:split_idx]
        
        return train, test
    
    def to_serializable(self) -> Dict[str, Any]:
        """Для сохранения в JSON."""
        # Синхронизация перед сохранением
        if self.is_double():
            self.fix_sync()
        
        return {
            "lottery_type": self.lottery_type,
            "blocks": self.blocks,
            "blocks2": self.blocks2,
            "metadata": self.metadata,
            "pick_count": self.pick_count,
            "total_numbers": self.total_numbers,
            "pick_count2": self.pick_count2,
            "total_numbers2": self.total_numbers2,
        }
    
    @classmethod
    def from_serializable(cls, data: Dict[str, Any]) -> 'LotteryData':
        obj = cls(data["lottery_type"])
        obj.blocks = data.get("blocks", [])
        obj.blocks2 = data.get("blocks2", [])
        obj.metadata = data.get("metadata", [])
        obj.pick_count = data.get("pick_count", 0)
        obj.total_numbers = data.get("total_numbers", 0)
        obj.pick_count2 = data.get("pick_count2", 0)
        obj.total_numbers2 = data.get("total_numbers2", 0)
        
        # Синхронизация после загрузки
        if obj.is_double():
            obj.fix_sync()
        
        return obj
    
    def get_statistics(self, field: int = 1) -> Dict[int, int]:
        """
        Возвращает частотную статистику для указанного поля.
        
        Args:
            field: 1 — первое поле, 2 — второе поле
        
        Returns:
            Словарь {число: частота}
        """
        if field == 1:
            blocks_to_use = self.blocks
        elif field == 2 and self.is_double():
            blocks_to_use = self.blocks2
        else:
            raise ValueError(f"Неверное поле {field}")
        
        counter = Counter()
        for block in blocks_to_use:
            counter.update(block)
        
        return dict(counter)
    
    def __len__(self) -> int:
        return self.blocks_count()
    
    def __repr__(self) -> str:
        sync_status = ""
        if self.is_double() and len(self.blocks) != len(self.blocks2):
            sync_status = " [НЕСИНХРОНИЗИРОВАН]"
        return f"<LotteryData type={self.lottery_type} blocks={self.blocks_count()}{sync_status} pick={self.pick_count}/{self.total_numbers}>"