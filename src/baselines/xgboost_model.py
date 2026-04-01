import numpy as np
import pandas as pd
import xgboost as xgb

def build_cross_sectional_features(returns: pd.DataFrame, lookback: int = 20,
                                    max_samples: int = 500_000
                                    ) -> tuple[np.ndarray, np.ndarray]:

    ret_vals = returns.values
    n_dates, n_assets = ret_vals.shape

    if n_dates <= lookback + 1:
        return np.empty((0, 6)), np.empty(0)

    ret_filled = np.nan_to_num(ret_vals, nan=0.0)

    all_X = []
    all_y = []

    valid_t_range = range(lookback, n_dates - 1)
    for t in valid_t_range:
        window = ret_vals[t - lookback:t]
        valid_count = np.sum(~np.isnan(window), axis=0)
        target = ret_vals[t]

        valid = (valid_count >= lookback // 2) & ~np.isnan(target)
        if valid.sum() < 5:
            continue

        w = np.nan_to_num(window, nan=0.0)

        mom_5 = w[-5:].sum(axis=0)
        mom_10 = w[-10:].sum(axis=0)
        mom_20 = w.sum(axis=0)
        vol_20 = np.nanstd(window, axis=0)
        vol_20 = np.nan_to_num(vol_20, nan=0.0)
        mean_20 = w.mean(axis=0)
        dist = w[-1] - mean_20
        ret_1d = w[-1]

        feats = np.column_stack([mom_5, mom_10, mom_20, vol_20, dist, ret_1d])

        valid_idx = np.where(valid)[0]
        all_X.append(feats[valid_idx])
        all_y.append(target[valid_idx])

    if not all_X:
        return np.empty((0, 6)), np.empty(0)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y

def build_features_single_step(returns_window: np.ndarray) -> np.ndarray:

    w = np.nan_to_num(returns_window, nan=0.0)
    T = w.shape[0]

    mom_5 = w[-5:].sum(axis=0) if T >= 5 else w.sum(axis=0)
    mom_10 = w[-10:].sum(axis=0) if T >= 10 else w.sum(axis=0)
    mom_20 = w.sum(axis=0)
    vol_20 = np.nanstd(returns_window, axis=0)
    vol_20 = np.nan_to_num(vol_20, nan=0.0)
    mean_20 = w.mean(axis=0)
    dist = w[-1] - mean_20
    ret_1d = w[-1]

    return np.column_stack([mom_5, mom_10, mom_20, vol_20, dist, ret_1d])

class XGBoostModel:

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
        self.xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
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

        self.model = xgb.XGBRegressor(**self.xgb_params)
        try:
            self.model.fit(X, y_clipped)
        except Exception:
            params = self.xgb_params.copy()
            params["device"] = "cpu"
            self.model = xgb.XGBRegressor(**params)
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
