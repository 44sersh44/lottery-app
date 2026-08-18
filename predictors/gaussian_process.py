from .base import BasePredictor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from core.features import extract_features
import numpy as np

class GaussianProcess(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None,
                 length_scale=1.0, random_state=42, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.length_scale = length_scale
        self.random_state = random_state + 3
        self.model = None
        self.scaler = StandardScaler()
        self.blocks = None

    def fit(self, blocks):
        self.blocks = blocks
        X, y = extract_features(blocks, self.total_numbers)
        if X.shape[0] == 0:
            return self
        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=1e-5)
        base_gp = GaussianProcessClassifier(
            kernel=kernel,
            random_state=self.random_state,
            max_iter_predict=100,
            warm_start=False
        )
        self.model = OneVsRestClassifier(base_gp)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_single(self):
        if self.blocks is None or len(self.blocks) == 0:
            return list(range(1, self.pick_count+1))
        last_block = self.blocks[-1]
        X_new, _ = extract_features([last_block], self.total_numbers)
        X_new_scaled = self.scaler.transform(X_new)
        proba = self.model.predict_proba(X_new_scaled)[0]
        if isinstance(proba, list):
            proba = np.mean(proba, axis=0)
        top_indices = np.argsort(proba)[-self.pick_count:][::-1]
        return sorted([i+1 for i in top_indices])