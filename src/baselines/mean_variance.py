import numpy as np
import pandas as pd

class MeanVarianceModel:

    def __init__(self, lookback: int = 252, max_weight: float = 0.05,
                 risk_aversion: float = 1.0, shrinkage: float = 0.5,
                 min_history: int = 60):

        self.lookback = lookback
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion
        self.shrinkage = shrinkage
        self.min_history = min_history

    def fit(self, train_data: dict) -> None:

        pass

    def _shrink_covariance(self, returns: np.ndarray) -> np.ndarray:

        sample_cov = np.cov(returns, rowvar=False)
        if sample_cov.ndim == 0:
            return np.array([[max(sample_cov, 1e-10)]])
        target = np.diag(np.diag(sample_cov))
        return (1 - self.shrinkage) * sample_cov + self.shrinkage * target

    def _solve_mv(self, mu: np.ndarray, cov: np.ndarray,
                   max_weight: float) -> np.ndarray:

        n = len(mu)
        gamma = self.risk_aversion

        cov_reg = cov + np.eye(n) * 1e-6

        try:

            raw_w = np.linalg.solve(cov_reg, mu) / gamma
        except np.linalg.LinAlgError:
            return np.ones(n) / n

        w = np.maximum(raw_w, 0.0)

        for _ in range(20):
            capped = w > max_weight
            if not capped.any():
                break
            excess = (w[capped] - max_weight).sum()
            w[capped] = max_weight
            uncapped = ~capped & (w > 0)
            if uncapped.sum() == 0:
                break
            w[uncapped] += excess / uncapped.sum()

        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n) / n

        return w

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        n_assets = returns.shape[1]

        lookback = min(self.lookback, len(returns))
        recent = returns.iloc[-lookback:]

        if "presence_mask" in current_data:
            mask = current_data["presence_mask"].iloc[-1].values.astype(bool)
        else:
            mask = ~np.isnan(returns.iloc[-1].values)

        has_history = recent.notna().sum().values >= self.min_history
        valid = mask & has_history
        valid_idx = np.where(valid)[0]

        weights = np.zeros(n_assets)
        if len(valid_idx) < 2:
            if len(valid_idx) == 1:
                weights[valid_idx[0]] = 1.0
            return weights

        valid_returns = recent.values[:, valid_idx]
        valid_returns = np.nan_to_num(valid_returns, nan=0.0)

        mu = valid_returns.mean(axis=0)
        cov = self._shrink_covariance(valid_returns)

        opt_w = self._solve_mv(mu, cov, self.max_weight)
        weights[valid_idx] = opt_w
        return weights
