import numpy as np
import pandas as pd
import lightgbm as lgb

from src.baselines.xgboost_model import build_cross_sectional_features, build_features_single_step

class LightGBMModel:

    def __init__(self, lookback: int = 20, top_k: int = 50,
                 max_weight: float = 0.05, n_estimators: int = 200,
                 max_depth: int = 4, learning_rate: float = 0.05,
                 min_history: int = 60, max_train_samples: int = 500_000):
        self.lookback = lookback
        self.top_k = top_k
        self.max_weight = max_weight
        self.min_history = min_history
        self.max_train_samples = max_train_samples
        self.model = None
        self.lgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "regression",
            "device": "gpu",
            "verbosity": -1,
            "n_jobs": -1,
        }

    def fit(self, train_data: dict) -> None:

        returns = train_data["returns"]
        X, y = build_cross_sectional_features(
            returns, self.lookback, self.max_train_samples
        )

        if len(X) < 100:
            self.model = None
            return

        y_clipped = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

        try:
            self.model = lgb.LGBMRegressor(**self.lgb_params)
            self.model.fit(X, y_clipped)
        except Exception:
            params = self.lgb_params.copy()
            params["device"] = "cpu"
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X, y_clipped)

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        n_assets = returns.shape[1]
        weights = np.zeros(n_assets)

        if self.model is None:
            mask = ~np.isnan(returns.iloc[-1].values)
            if mask.sum() > 0:
                weights[mask] = 1.0 / mask.sum()
            return weights

        lookback = min(self.lookback, len(returns))
        window = returns.iloc[-lookback:].values
        feats = build_features_single_step(window)

        if "presence_mask" in current_data:
            mask = current_data["presence_mask"].iloc[-1].values.astype(bool)
        else:
            mask = ~np.isnan(returns.iloc[-1].values)

        valid_count = np.sum(~np.isnan(window), axis=0)
        has_history = valid_count >= (lookback // 2)
        valid = mask & has_history

        if valid.sum() == 0:
            return weights

        valid_idx = np.where(valid)[0]
        preds = self.model.predict(feats[valid_idx])

        k = min(self.top_k, len(valid_idx))
        top_indices = np.argsort(preds)[-k:]
        selected = valid_idx[top_indices]

        w_val = min(1.0 / k, self.max_weight)
        weights[selected] = w_val
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights
