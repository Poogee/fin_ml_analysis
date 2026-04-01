import numpy as np
import pandas as pd

class EqualWeightModel:

    def fit(self, train_data: dict) -> None:

        pass

    def predict_weights(self, current_data: dict) -> np.ndarray:

        n_assets = current_data["returns"].shape[1]

        if "presence_mask" in current_data:
            mask = current_data["presence_mask"].iloc[-1].values.astype(bool)
        else:

            mask = ~np.isnan(current_data["returns"].iloc[-1].values)

        weights = np.zeros(n_assets)
        n_available = mask.sum()
        if n_available > 0:
            weights[mask] = 1.0 / n_available
        return weights
