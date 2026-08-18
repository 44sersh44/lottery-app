# services/data_service.py
from typing import Optional
from core.lottery_data import LotteryData
from data_io.csv_loader import CSVLoader
from services.profile_manager import ProfileManager

class DataService:
    def __init__(self, profile_manager: ProfileManager):
        self.profile_manager = profile_manager

    @property
    def current_data(self) -> Optional[LotteryData]:
        """Возвращает данные активного профиля."""
        if self.profile_manager.current_profile:
            return self.profile_manager.current_profile.data
        return None

    @property
    def current_profile_path(self) -> Optional[str]:
        if self.profile_manager.current_profile:
            return self.profile_manager.current_profile.profile_path
        return None

    def load_csv(self, filepath: str, replace: bool = False) -> tuple:
        """Загружает CSV в текущий профиль."""
        if not self.current_data:
            raise ValueError("Нет активного профиля")
        count, errors = CSVLoader.load(filepath, self.current_data, replace=replace, sort_by_date=False)
        self.profile_manager.current_profile.save()
        return count, errors

    def get_latest_block(self):
        if self.current_data:
            return self.current_data.get_latest_block()
        return None

    def get_blocks(self, field: int = 1):
        if not self.current_data:
            return []
        if field == 1:
            return self.current_data.blocks
        else:
            return self.current_data.blocks2

    def get_metadata(self):
        if self.current_data:
            return self.current_data.metadata
        return []

    def is_double(self) -> bool:
        if self.current_data:
            return self.current_data.is_double()
        return False