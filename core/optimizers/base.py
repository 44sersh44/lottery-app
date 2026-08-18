class BaseOptimizer:
    """Базовый класс для всех оптимизаторов."""

    def __init__(self, tuner):
        """
        :param tuner: экземпляр HyperparameterTuner для доступа к его методам.
        """
        self.tuner = tuner

    def optimize(self, model_name: str, param_grid: dict, all_blocks: list,
                 initial_window: int, pick_count: int, total_numbers: int,
                 field: int, **kwargs) -> dict:
        """
        Запускает оптимизацию.
        Должен вернуть словарь с ключами:
            'best_params': dict,
            'best_score': float,
            'best_avg_matches': float,
            'matches_list': list,
            'total_iterations': int,
            'iterations_file': str or None
        """
        raise NotImplementedError