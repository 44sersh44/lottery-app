"""Управление профилями — сохранение, загрузка, история прогнозов."""
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.lottery_data import LotteryData


class LotteryProfile:
    """Профиль лотереи: хранит данные, параметры и историю прогнозов."""
    
    # Максимальное количество записей в истории
    MAX_HISTORY_SIZE = 100

    def __init__(self, profile_path: str, name: str, auto_save: bool = True):
        """
        Args:
            profile_path: Путь к папке профиля
            name: Имя профиля
            auto_save: Автоматически сохранять при изменениях
        """
        self.name = name
        self.profile_path = profile_path
        self.auto_save = auto_save
        self.data = LotteryData()
        self.predictions_history: List[Dict] = []

        # Загружаем существующие данные, если есть
        self._load()

    @staticmethod
    def _convert_to_serializable(obj):
        """Преобразует numpy типы в Python типы для JSON."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: LotteryProfile._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LotteryProfile._convert_to_serializable(i) for i in obj]
        return obj

    def _load(self):
        """Загрузить профиль из файлов с обработкой ошибок."""
        os.makedirs(self.profile_path, exist_ok=True)
        data_file = os.path.join(self.profile_path, "data.json")
        history_file = os.path.join(self.profile_path, "history.json")

        # Загрузка данных
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    self.data = LotteryData.from_serializable(data_dict)
                    print(f"[LOAD] Загружен профиль '{self.name}': {self.data.blocks_count()} блоков")
            except json.JSONDecodeError as e:
                print(f"[ERR] Ошибка чтения {data_file}: {e}")
                print(f"[WARN] Создаётся новый файл данных")
                self.data = LotteryData()
            except Exception as e:
                print(f"[ERR] Неожиданная ошибка при загрузке {data_file}: {e}")
                self.data = LotteryData()

        # Загрузка истории
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.predictions_history = json.load(f)
                # Ограничиваем размер истории
                if len(self.predictions_history) > self.MAX_HISTORY_SIZE:
                    self.predictions_history = self.predictions_history[-self.MAX_HISTORY_SIZE:]
                    print(f"[LOAD] История прогнозов ограничена {self.MAX_HISTORY_SIZE} записями")
            except json.JSONDecodeError as e:
                print(f"[ERR] Ошибка чтения {history_file}: {e}")
                print(f"[WARN] Создаётся новая история")
                self.predictions_history = []
            except Exception as e:
                print(f"[ERR] Неожиданная ошибка при загрузке {history_file}: {e}")
                self.predictions_history = []

    def save(self) -> bool:
        """
        Сохранить профиль с преобразованием numpy типов.
        
        Returns:
            True при успехе, False при ошибке
        """
        os.makedirs(self.profile_path, exist_ok=True)
        data_file = os.path.join(self.profile_path, "data.json")
        history_file = os.path.join(self.profile_path, "history.json")
        data_tmp = data_file + ".tmp"
        history_tmp = history_file + ".tmp"
        
        try:
            # Синхронизируем данные перед сохранением
            if self.data.is_double():
                self.data.fix_sync()
            
            # Сохраняем данные
            with open(data_tmp, 'w', encoding='utf-8') as f:
                json.dump(
                    self._convert_to_serializable(self.data.to_serializable()), 
                    f, ensure_ascii=False, indent=2
                )
            
            # Сохраняем историю с преобразованием
            with open(history_tmp, 'w', encoding='utf-8') as f:
                json.dump(
                    self._convert_to_serializable(self.predictions_history), 
                    f, ensure_ascii=False, indent=2
                )
            
            # Атомарная замена
            if os.path.exists(data_file):
                os.replace(data_tmp, data_file)
            else:
                os.rename(data_tmp, data_file)
            
            if os.path.exists(history_file):
                os.replace(history_tmp, history_file)
            else:
                os.rename(history_tmp, history_file)
            
            return True
                
        except Exception as e:
            print(f"[ERR] Ошибка сохранения профиля '{self.name}': {e}")
            for tmp in [data_tmp, history_tmp]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except:
                        pass
            return False

    def delete(self, confirm: bool = True) -> bool:
        """
        Удалить профиль с диска.
        
        Args:
            confirm: Запрашивать подтверждение
        
        Returns:
            True при успехе, False при ошибке или отмене
        """
        if confirm:
            response = input(f"Удалить профиль '{self.name}'? (да/нет): ").strip().lower()
            if response not in ['да', 'yes', '1', 'д', '+']:
                print("[INFO] Удаление отменено")
                return False
        
        if os.path.exists(self.profile_path):
            try:
                shutil.rmtree(self.profile_path)
                print(f"[OK] Профиль '{self.name}' удалён")
                return True
            except Exception as e:
                print(f"[ERR] Ошибка удаления профиля '{self.name}': {e}")
                return False
        return False

    def backup(self, backup_path: str = None) -> bool:
        """
        Создаёт резервную копию профиля.
        
        Args:
            backup_path: Путь для резервной копии (если None, создаётся автоматически)
        
        Returns:
            True при успехе, False при ошибке
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.profile_path}_backup_{timestamp}"
        
        try:
            shutil.copytree(self.profile_path, backup_path)
            print(f"[OK] Резервная копия создана: {backup_path}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка создания резервной копии: {e}")
            return False

    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Восстанавливает профиль из резервной копии.
        
        Args:
            backup_path: Путь к резервной копии
        
        Returns:
            True при успехе, False при ошибке
        """
        if not os.path.exists(backup_path):
            print(f"[ERR] Резервная копия не найдена: {backup_path}")
            return False
        
        try:
            # Удаляем текущий профиль
            if os.path.exists(self.profile_path):
                shutil.rmtree(self.profile_path)
            
            # Копируем резервную копию
            shutil.copytree(backup_path, self.profile_path)
            
            # Перезагружаем данные
            self._load()
            print(f"[OK] Профиль восстановлен из резервной копии: {backup_path}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка восстановления из резервной копии: {e}")
            return False

    def add_prediction_to_history(self, prediction: Dict) -> None:
        """
        Добавить прогноз в историю с автоматическим ограничением размера.
        
        Args:
            prediction: Словарь с ключами:
                - timestamp: время прогноза
                - prediction: список чисел (поле1)
                - prediction2: список чисел (поле2, опционально)
                - confidence: уверенность в процентах
                - type: тип прогноза (ensemble, positional и т.д.)
        """
        # Добавляем timestamp, если его нет
        if 'timestamp' not in prediction:
            prediction['timestamp'] = datetime.now().isoformat()
        
        self.predictions_history.append(prediction)
        
        # Ограничиваем размер истории
        if len(self.predictions_history) > self.MAX_HISTORY_SIZE:
            removed = len(self.predictions_history) - self.MAX_HISTORY_SIZE
            self.predictions_history = self.predictions_history[-self.MAX_HISTORY_SIZE:]
            print(f"[HISTORY] Удалено {removed} старых прогнозов (лимит {self.MAX_HISTORY_SIZE})")
        
        # Автосохранение после добавления
        if self.auto_save:
            self.save()

    def get_prediction_history(self, limit: int = None) -> List[Dict]:
        """
        Получить историю прогнозов.
        
        Args:
            limit: Максимальное количество записей (None = все)
        
        Returns:
            Список прогнозов (от свежих к старым)
        """
        history = list(reversed(self.predictions_history))  # свежие первыми
        if limit:
            history = history[:limit]
        return history

    def clear_history(self) -> None:
        """Очистить историю прогнозов."""
        self.predictions_history = []
        if self.auto_save:
            self.save()
        print(f"[OK] История прогнозов профиля '{self.name}' очищена")

    def export_to_json(self, export_path: str) -> bool:
        """
        Экспортировать профиль в JSON файл.
        
        Args:
            export_path: Путь для сохранения
        
        Returns:
            True при успехе, False при ошибке
        """
        try:
            export_data = {
                "profile_name": self.name,
                "export_date": datetime.now().isoformat(),
                "data": self.data.to_serializable(),
                "predictions_history": self.predictions_history
            }
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Профиль экспортирован в {export_path}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка экспорта: {e}")
            return False

    def import_from_json(self, import_path: str, merge: bool = True) -> bool:
        """
        Импортировать профиль из JSON файла.
        
        Args:
            import_path: Путь к файлу для импорта
            merge: Если True, объединяет с существующими данными
        
        Returns:
            True при успехе, False при ошибке
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if merge:
                # Объединение данных
                if "data" in import_data:
                    imported_data = LotteryData.from_serializable(import_data["data"])
                    
                    # Получаем существующие ID тиражей
                    existing_ids = set()
                    for meta in self.data.metadata:
                        existing_ids.add(meta.get('draw_id'))
                    
                    # Добавляем новые блоки
                    added = 0
                    for i in range(imported_data.blocks_count()):
                        draw_id = imported_data.metadata[i].get('draw_id')
                        if draw_id not in existing_ids:
                            if imported_data.is_double():
                                self.data.add_block(
                                    (imported_data.blocks[i], imported_data.blocks2[i]),
                                    imported_data.metadata[i]
                                )
                            else:
                                self.data.add_block(imported_data.blocks[i], imported_data.metadata[i])
                            added += 1
                    
                    print(f"[IMPORT] Добавлено {added} новых блоков")
                
                if "predictions_history" in import_data:
                    # Объединяем историю прогнозов
                    existing_timestamps = {p.get('timestamp') for p in self.predictions_history}
                    for pred in import_data["predictions_history"]:
                        if pred.get('timestamp') not in existing_timestamps:
                            self.predictions_history.append(pred)
                    
                    # Ограничиваем размер
                    if len(self.predictions_history) > self.MAX_HISTORY_SIZE:
                        self.predictions_history = self.predictions_history[-self.MAX_HISTORY_SIZE:]
            else:
                # Полная замена
                if "data" in import_data:
                    self.data = LotteryData.from_serializable(import_data["data"])
                
                if "predictions_history" in import_data:
                    self.predictions_history = import_data["predictions_history"][:self.MAX_HISTORY_SIZE]
            
            if self.auto_save:
                self.save()
            print(f"[OK] Профиль импортирован из {import_path}")
            return True
        except Exception as e:
            print(f"[ERR] Ошибка импорта: {e}")
            return False

    # --- Улучшенные свойства с проверками ---
    
    @property
    def blocks(self):
        return self.data.blocks

    @blocks.setter
    def blocks(self, value):
        if self.data.is_double():
            # При установке blocks для double, синхронизируем с blocks2
            old_len = len(self.data.blocks)
            self.data.blocks = value
            if len(self.data.blocks2) != len(value):
                # Пытаемся восстановить синхронизацию
                self.data.fix_sync()
                print(f"[WARN] При установке blocks синхронизация восстановлена")
        else:
            self.data.blocks = value
        if self.auto_save:
            self.save()

    @property
    def blocks1(self):
        return self.data.blocks

    @blocks1.setter
    def blocks1(self, value):
        self.blocks = value  # используем основной сеттер

    @property
    def blocks2(self):
        return self.data.blocks2

    @blocks2.setter
    def blocks2(self, value):
        self.data.blocks2 = value
        if self.data.is_double() and len(self.data.blocks) != len(value):
            self.data.fix_sync()
            print(f"[WARN] При установке blocks2 синхронизация восстановлена")
        if self.auto_save:
            self.save()

    @property
    def blocks_metadata(self):
        return self.data.metadata

    @blocks_metadata.setter
    def blocks_metadata(self, value):
        self.data.metadata = value
        if self.auto_save:
            self.save()

    # --- Остальные свойства ---
    
    @property
    def lottery_type(self):
        return self.data.lottery_type

    @lottery_type.setter
    def lottery_type(self, value):
        self.data.lottery_type = value
        if self.auto_save:
            self.save()

    @property
    def pick_count(self):
        return self.data.pick_count

    @pick_count.setter
    def pick_count(self, value):
        self.data.pick_count = value
        if self.auto_save:
            self.save()

    @property
    def total_numbers(self):
        return self.data.total_numbers

    @total_numbers.setter
    def total_numbers(self, value):
        self.data.total_numbers = value
        if self.auto_save:
            self.save()

    @property
    def pick_count1(self):
        return self.data.pick_count

    @pick_count1.setter
    def pick_count1(self, value):
        self.data.pick_count = value
        if self.auto_save:
            self.save()

    @property
    def total_numbers1(self):
        return self.data.total_numbers

    @total_numbers1.setter
    def total_numbers1(self, value):
        self.data.total_numbers = value
        if self.auto_save:
            self.save()

    @property
    def pick_count2(self):
        return self.data.pick_count2

    @pick_count2.setter
    def pick_count2(self, value):
        self.data.pick_count2 = value
        if self.auto_save:
            self.save()

    @property
    def total_numbers2(self):
        return self.data.total_numbers2

    @total_numbers2.setter
    def total_numbers2(self, value):
        self.data.total_numbers2 = value
        if self.auto_save:
            self.save()

    @property
    def sequence_length(self):
        # для совместимости
        return 3

    @sequence_length.setter
    def sequence_length(self, value):
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику профиля."""
        stats = {
            "name": self.name,
            "type": self.data.lottery_type,
            "blocks_count": self.data.blocks_count(),
            "history_size": len(self.predictions_history),
            "has_blocks2": len(self.data.blocks2) > 0 if self.data.is_double() else False,
        }
        
        if self.data.blocks_count() > 0:
            stats["latest_block"] = self.data.get_latest_block()
        
        return stats

    def __repr__(self) -> str:
        return f"<LotteryProfile name={self.name} blocks={self.data.blocks_count()} history={len(self.predictions_history)}>"