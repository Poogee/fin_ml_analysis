import numpy as np
import pandas as pd

class RiskParityModel:

    def __init__(self, vol_lookback: int = 60, min_history: int = 20):

        self.vol_lookback = vol_lookback
        self.min_history = min_history

    def fit(self, train_data: dict) -> None:

        pass

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        n_assets = returns.shape[1]

        lookback = min(self.vol_lookback, len(returns))
        recent_returns = returns.iloc[-lookback:]

        vols = recent_returns.std().values

        if "presence_mask" in current_data:
            mask = current_data["presence_mask"].iloc[-1].values.astype(bool)
        else:
            mask = ~np.isnan(returns.iloc[-1].values)

        has_history = recent_returns.notna().sum().values >= self.min_history
        has_vol = (vols > 1e-10)
        valid = mask & has_history & has_vol

        weights = np.zeros(n_assets)
        if valid.sum() == 0:
            return weights

        inv_vol = np.where(valid, 1.0 / vols, 0.0)
        weights = inv_vol / inv_vol.sum()
        return weights
