# services/profile_manager.py
import os
from typing import Optional, Dict
from core.profile import LotteryProfile

class ProfileManager:
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = profiles_dir
        self.profiles: Dict[str, LotteryProfile] = {}
        self.current_profile: Optional[LotteryProfile] = None
        os.makedirs(profiles_dir, exist_ok=True)
        self._load_profiles()

    def _load_profiles(self):
        """Загружает все профили из папки."""
        for item in os.listdir(self.profiles_dir):
            path = os.path.join(self.profiles_dir, item)
            if os.path.isdir(path):
                try:
                    profile = LotteryProfile(path, item)
                    self.profiles[item] = profile
                except Exception as e:
                    print(f"Ошибка загрузки профиля {item}: {e}")

    def create_profile(self, name: str, lottery_type: str = "single",
                       pick_count: int = 6, total_numbers: int = 45,
                       pick_count2: Optional[int] = None,
                       total_numbers2: Optional[int] = None) -> LotteryProfile:
        """Создаёт новый профиль с заданными параметрами."""
        if name in self.profiles:
            raise ValueError(f"Профиль '{name}' уже существует")
        profile_path = os.path.join(self.profiles_dir, name)
        os.makedirs(profile_path, exist_ok=True)
        profile = LotteryProfile(profile_path, name)
        profile.data.lottery_type = lottery_type
        profile.data.pick_count = pick_count
        profile.data.total_numbers = total_numbers
        if lottery_type == "double":
            if pick_count2 is None or total_numbers2 is None:
                raise ValueError("Для двойной лотереи нужны параметры второго поля")
            profile.data.pick_count2 = pick_count2
            profile.data.total_numbers2 = total_numbers2
        profile.save()
        self.profiles[name] = profile
        self.current_profile = profile
        return profile

    def select_profile(self, name: str) -> Optional[LotteryProfile]:
        """Выбирает профиль по имени."""
        if name not in self.profiles:
            raise ValueError(f"Профиль '{name}' не найден")
        self.current_profile = self.profiles[name]
        return self.current_profile

    def delete_profile(self, name: str):
        """Удаляет профиль."""
        if name not in self.profiles:
            raise ValueError(f"Профиль '{name}' не найден")
        self.profiles[name].delete()
        del self.profiles[name]
        if self.current_profile and self.current_profile.name == name:
            self.current_profile = None

    def get_profile_names(self) -> list:
        return list(self.profiles.keys())

    def get_profile_info(self, name: str) -> dict:
        """Возвращает информацию о профиле для отображения."""
        profile = self.profiles.get(name)
        if not profile:
            return {}
        data = profile.data
        return {
            'name': name,
            'type': data.lottery_type,
            'blocks': data.blocks_count(),
            'pick_count': data.pick_count,
            'total_numbers': data.total_numbers,
            'pick_count2': data.pick_count2 if data.is_double() else None,
            'total_numbers2': data.total_numbers2 if data.is_double() else None
        }