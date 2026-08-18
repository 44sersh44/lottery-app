from .base import BasePredictor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from core.features import extract_features
import numpy as np

class RBF(BasePredictor):
    def __init__(self, total_numbers=None, pick_count=None,
                 C=1.0, gamma='scale', random_state=42, **kwargs):
        super().__init__(total_numbers=total_numbers, pick_count=pick_count, **kwargs)
        self.C = C
        self.gamma = gamma
        self.random_state = random_state + 2
        self.model = None
        self.scaler = StandardScaler()
        self.blocks = None

    def fit(self, blocks):
        self.blocks = blocks
        X, y = extract_features(blocks, self.total_numbers)
        if X.shape[0] == 0:
            return self
        base_svc = SVC(kernel='rbf', C=self.C, gamma=self.gamma,
                       probability=True, random_state=self.random_state)
        self.model = OneVsRestClassifier(base_svc)
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