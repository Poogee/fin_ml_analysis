import numpy as np

class SmoothedModelWrapper:

    def __init__(self, model, alpha: float = 0.3):

        self.model = model
        self.alpha = alpha
        self._prev_weights = None

    def fit(self, train_data: dict) -> None:
        self.model.fit(train_data)
        self._prev_weights = None

    def predict_weights(self, current_data: dict) -> np.ndarray:
        raw_weights = self.model.predict_weights(current_data)

        if self._prev_weights is None or len(self._prev_weights) != len(raw_weights):
            self._prev_weights = raw_weights.copy()
            return raw_weights

        smoothed = self.alpha * raw_weights + (1 - self.alpha) * self._prev_weights

        w_sum = smoothed.sum()
        if w_sum > 0:
            smoothed /= w_sum

        self._prev_weights = smoothed.copy()
        return smoothed

class RankWeightedWrapper:

    def __init__(self, model, temperature: float = 1.0, max_weight: float = 0.05):
        self.model = model
        self.temperature = temperature
        self.max_weight = max_weight
        self._prev_weights = None

    def fit(self, train_data: dict) -> None:
        self.model.fit(train_data)
        self._prev_weights = None

    def predict_weights(self, current_data: dict) -> np.ndarray:

        raw = self.model.predict_weights(current_data)

        if self._prev_weights is not None and len(self._prev_weights) == len(raw):
            raw = 0.4 * raw + 0.6 * self._prev_weights
            w_sum = raw.sum()
            if w_sum > 0:
                raw /= w_sum

        self._prev_weights = raw.copy()
        return raw
