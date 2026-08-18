class BasePredictor:
    """Базовый класс для всех предикторов."""
    def __init__(self, total_numbers, pick_count, **kwargs):
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.params = kwargs

    def fit(self, blocks):
        raise NotImplementedError

    def predict_single(self):
        raise NotImplementedError
