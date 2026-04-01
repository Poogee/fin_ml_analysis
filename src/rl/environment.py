from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,
        features_per_asset: np.ndarray | None = None,
        features_global: np.ndarray | None = None,
        lookback: int = 20,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        reward_type: str = "return",
        reward_window: int = 20,
        lambda_tc: float = 1.0,
        lambda_dd: float = 2.0,
        lambda_turnover: float = 0.5,
        dd_threshold: float = 0.05,
        dsr_eta: float = 0.005,
        max_weight: float = 0.1,
        reward_clip: tuple[float, float] | None = None,
        turbulence_threshold_pct: float | None = None,
    ):
        super().__init__()

        self.returns = np.nan_to_num(returns, nan=0.0).astype(np.float32)
        self.n_dates, self.n_assets = self.returns.shape
        self.lookback = lookback
        self.tc_rate = (transaction_cost_bps + slippage_bps) / 10000.0
        self.reward_type = reward_type
        self.reward_window = reward_window
        self.lambda_tc = lambda_tc
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.dd_threshold = dd_threshold
        self.dsr_eta = dsr_eta
        self.max_weight = max_weight
        self.reward_clip = reward_clip
        self.turbulence_threshold_pct = turbulence_threshold_pct

        self._turbulence_threshold = None
        if turbulence_threshold_pct is not None:
            self._turbulence_threshold = self._compute_turbulence_threshold(
                returns, turbulence_threshold_pct,
            )

        if features_per_asset is not None:
            self.features_pa = np.nan_to_num(features_per_asset, nan=0.0).astype(np.float32)
            self.n_pa_features = self.features_pa.shape[2]
        else:
            self.features_pa = None
            self.n_pa_features = 0

        if features_global is not None:
            self.features_global = np.nan_to_num(features_global, nan=0.0).astype(np.float32)
            self.n_global_features = self.features_global.shape[1]
        else:
            self.features_global = None
            self.n_global_features = 0

        self.obs_per_asset_dim = self.lookback * (1 + self.n_pa_features)
        self.obs_dim = (
            self.n_assets * self.obs_per_asset_dim
            + self.n_assets
            + self.n_global_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        self._step = 0
        self._weights = np.zeros(self.n_assets, dtype=np.float32)
        self._portfolio_returns = []

        self._dsr_A = 0.0
        self._dsr_B = 0.0

        self._peak_value = 1.0
        self._cumulative_value = 1.0

    @staticmethod
    def _compute_turbulence_threshold(
        returns: np.ndarray, percentile: float, window: int = 60,
    ) -> float:

        T, N = returns.shape
        turbulence_values = []
        for t in range(window, T):
            w = returns[t - window:t]
            mu = w.mean(axis=0)
            cov = np.cov(w, rowvar=False) + np.eye(N) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
            diff = returns[t] - mu
            turb = float(diff @ cov_inv @ diff) / N
            turbulence_values.append(turb)
        if not turbulence_values:
            return float("inf")
        return float(np.percentile(turbulence_values, percentile))

    def _is_turbulent(self, t: int, window: int = 60) -> bool:

        if self._turbulence_threshold is None:
            return False
        if t < window:
            return False
        w = self.returns[t - window:t]
        N = self.n_assets
        mu = w.mean(axis=0)
        cov = np.cov(w, rowvar=False) + np.eye(N) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return False
        diff = self.returns[t] - mu
        turb = float(diff @ cov_inv @ diff) / N
        return turb > self._turbulence_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = self.lookback
        self._weights = np.zeros(self.n_assets, dtype=np.float32)
        self._portfolio_returns = []
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._peak_value = 1.0
        self._cumulative_value = 1.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):

        new_weights = self._normalize_weights(action)

        is_turbulent = self._is_turbulent(self._step)
        if is_turbulent:
            new_weights = np.zeros(self.n_assets, dtype=np.float32)

        turnover = np.abs(new_weights - self._weights).sum()
        tc = turnover * self.tc_rate

        day_returns = self.returns[self._step]
        port_return = np.sum(new_weights * day_returns) - tc

        self._portfolio_returns.append(port_return)

        self._cumulative_value *= (1.0 + port_return)
        self._peak_value = max(self._peak_value, self._cumulative_value)
        current_drawdown = 1.0 - self._cumulative_value / self._peak_value

        self._weights = new_weights

        reward = self._compute_reward(port_return, tc, turnover, current_drawdown)

        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])

        self._step += 1
        terminated = self._step >= self.n_dates
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_return": port_return,
            "transaction_cost": tc,
            "turnover": turnover,
            "drawdown": current_drawdown,
            "turbulent": is_turbulent,
        }

        return obs, reward, terminated, truncated, info

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        w = np.clip(action, 0.0, None).astype(np.float32)

        w = np.minimum(w, self.max_weight)

        total = w.sum()
        if total > 0:
            w = w / total
        else:

            w = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        return w

    def _get_obs(self) -> np.ndarray:

        t = self._step
        start = max(0, t - self.lookback)
        actual_len = t - start

        ret_window = self.returns[start:t]
        if actual_len < self.lookback:
            pad = np.zeros((self.lookback - actual_len, self.n_assets), dtype=np.float32)
            ret_window = np.concatenate([pad, ret_window])

        if self.features_pa is not None:

            pa_window = self.features_pa[start:t]
            if actual_len < self.lookback:
                pad_pa = np.zeros((self.lookback - actual_len, self.n_assets, self.n_pa_features), dtype=np.float32)
                pa_window = np.concatenate([pad_pa, pa_window])

            ret_flat = ret_window.T
            pa_flat = pa_window.transpose(1, 0, 2).reshape(self.n_assets, -1)
            per_asset = np.concatenate([ret_flat, pa_flat], axis=1).flatten()
        else:

            per_asset = ret_window.T.flatten()

        parts = [per_asset, self._weights]

        if self.features_global is not None and t < len(self.features_global):
            parts.append(self.features_global[t])
        elif self.n_global_features > 0:
            parts.append(np.zeros(self.n_global_features, dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    def _compute_dsr(self, port_return: float) -> float:

        eta = self.dsr_eta
        dA = port_return - self._dsr_A
        dB = port_return ** 2 - self._dsr_B

        denom = self._dsr_B - self._dsr_A ** 2
        if denom > 1e-12:
            dsr = (self._dsr_B * dA - 0.5 * self._dsr_A * dB) / (denom ** 1.5)
        else:
            dsr = port_return * 100.0

        self._dsr_A = eta * port_return + (1 - eta) * self._dsr_A
        self._dsr_B = eta * port_return ** 2 + (1 - eta) * self._dsr_B

        return float(dsr)

    def _compute_reward(self, port_return: float, tc: float,
                        turnover: float = 0.0,
                        current_drawdown: float = 0.0) -> float:

        if self.reward_type == "dsr":

            return self._compute_dsr(port_return)

        elif self.reward_type == "composite":

            dsr = self._compute_dsr(port_return)

            dd_penalty = max(0.0, current_drawdown - self.dd_threshold)

            turnover_penalty = turnover

            reward = (dsr
                      - self.lambda_dd * dd_penalty
                      - self.lambda_turnover * turnover_penalty)
            return reward

        elif self.reward_type == "risk_sensitive":

            log_ret = float(np.log1p(port_return))

            calmar_component = 0.0
            if current_drawdown > 1e-8:
                window = self._portfolio_returns[-self.reward_window:]
                ann_ret = np.mean(window) * 252 if len(window) >= 5 else 0.0
                calmar_component = ann_ret / current_drawdown
                calmar_component = np.clip(calmar_component, -5.0, 5.0)

            vol_component = 0.0
            window = self._portfolio_returns[-self.reward_window:]
            if len(window) >= 5:
                vol_component = np.std(window) * np.sqrt(252)
            reward = log_ret * 100.0 + 0.1 * calmar_component - 0.05 * vol_component
            return reward

        elif self.reward_type == "log_return":

            return float(np.log1p(port_return)) * 100.0

        elif self.reward_type == "sharpe":

            window = self._portfolio_returns[-self.reward_window:]
            if len(window) < 5:
                return port_return
            mean_r = np.mean(window)
            std_r = np.std(window)
            if std_r < 1e-8:
                return mean_r * 100
            return (mean_r / std_r) * np.sqrt(252)

        else:

            return port_return - self.lambda_tc * tc

class DiscretePortfolioEnv(PortfolioEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_discrete_actions = self.n_assets + 3
        self.action_space = spaces.Discrete(self.n_discrete_actions)

    def step(self, action: int):

        weights = self._discrete_to_weights(action)
        return super().step(weights)

    def _discrete_to_weights(self, action: int) -> np.ndarray:

        equal = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        if action == 0:
            return equal

        elif 1 <= action <= self.n_assets:

            w = equal * 0.5
            asset_idx = action - 1
            w[asset_idx] += 0.5
            return w / w.sum()

        elif action == self.n_assets + 1:

            t = self._step
            if t >= 5:
                recent = self.returns[t - 5:t].mean(axis=0)
                scores = np.maximum(recent, 0)
                if scores.sum() > 0:
                    return scores / scores.sum()
            return equal

        elif action == self.n_assets + 2:

            t = self._step
            if t >= 5:
                recent = self.returns[t - 5:t].mean(axis=0)
                scores = np.maximum(-recent, 0)
                if scores.sum() > 0:
                    return scores / scores.sum()
            return equal

        return equal

class LongShortPortfolioEnv(PortfolioEnv):

    def __init__(
        self,
        strategy: str = "130_30",
        max_long_weight: float = 0.10,
        max_short_weight: float = 0.05,
        max_gross_exposure: float = 1.6,
        annual_borrow_rate: float = 0.005,
        short_tc_premium_bps: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strategy = strategy
        self.max_long_weight = max_long_weight
        self.max_short_weight = max_short_weight
        self.max_gross_exposure = max_gross_exposure
        self.daily_borrow_rate = annual_borrow_rate / 252
        self.short_tc_premium = short_tc_premium_bps / 10000

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        w = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self.strategy == "130_30":
            return self._normalize_130_30(w)
        elif self.strategy == "dollar_neutral":
            return self._normalize_dollar_neutral(w)
        return self._normalize_flexible(w)

    def _normalize_130_30(self, w: np.ndarray) -> np.ndarray:

        w = np.clip(w, -self.max_short_weight, self.max_long_weight)

        long_mask = w > 0
        short_mask = w < 0

        long_sum = w[long_mask].sum() if long_mask.any() else 0
        short_sum = abs(w[short_mask].sum()) if short_mask.any() else 0

        if long_sum > 0:
            w[long_mask] *= 1.3 / long_sum
        else:
            w = np.ones_like(w) / len(w) * 1.3

        if short_sum > 0:
            w[short_mask] *= 0.3 / short_sum

        w = np.clip(w, -self.max_short_weight, self.max_long_weight)
        return w

    def _normalize_dollar_neutral(self, w: np.ndarray) -> np.ndarray:

        w = np.clip(w, -self.max_short_weight, self.max_long_weight)
        w = w - w.mean()
        gross = np.abs(w).sum()
        if gross > 0:
            w *= self.max_gross_exposure / gross
        w = np.clip(w, -self.max_short_weight, self.max_long_weight)
        return w

    def _normalize_flexible(self, w: np.ndarray) -> np.ndarray:

        w = np.clip(w, -self.max_short_weight, self.max_long_weight)
        gross = np.abs(w).sum()
        if gross > self.max_gross_exposure:
            w *= self.max_gross_exposure / gross
        return w

    def step(self, action: np.ndarray):
        new_weights = self._normalize_weights(action)

        weight_changes = new_weights - self._weights
        long_changes = np.abs(weight_changes[weight_changes > 0]).sum()
        short_changes = np.abs(weight_changes[weight_changes < 0]).sum()
        tc = long_changes * self.tc_rate + short_changes * (self.tc_rate + self.short_tc_premium)

        short_exposure = abs(np.minimum(new_weights, 0).sum())
        borrow_cost = short_exposure * self.daily_borrow_rate

        day_returns = self.returns[self._step]
        port_return = np.sum(new_weights * day_returns) - tc - borrow_cost

        self._portfolio_returns.append(port_return)

        self._cumulative_value *= (1.0 + port_return)
        self._peak_value = max(self._peak_value, self._cumulative_value)
        current_drawdown = 1.0 - self._cumulative_value / self._peak_value

        turnover = np.abs(new_weights - self._weights).sum()
        self._weights = new_weights

        reward = self._compute_reward(port_return, tc + borrow_cost, turnover, current_drawdown)

        self._step += 1
        terminated = self._step >= self.n_dates
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_return": port_return,
            "transaction_cost": tc,
            "borrow_cost": borrow_cost,
            "turnover": turnover,
            "drawdown": current_drawdown,
            "short_exposure": short_exposure,
        }

        return obs, reward, terminated, truncated, info

class SAPPOEnv(PortfolioEnv):

    def __init__(self, sappo_alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.sappo_alpha = sappo_alpha

    def step(self, action: np.ndarray):
        prev_weights = self._weights.copy()

        obs, reward, terminated, truncated, info = super().step(action)

        new_weights = self._weights
        weight_change = new_weights - prev_weights

        t = self._step - 1
        sentiment_signal = self._get_sentiment_signal(t)

        if sentiment_signal is not None:
            alignment = weight_change * sentiment_signal
            alignment_mean = float(alignment.mean())
            reward = reward * (1.0 + self.sappo_alpha * alignment_mean)
            info["sappo_alignment"] = alignment_mean

        return obs, reward, terminated, truncated, info

    def _get_sentiment_signal(self, t: int) -> np.ndarray | None:

        if self.features_pa is not None and t < len(self.features_pa):

            signal = self.features_pa[t, :, -1].copy()

            max_abs = np.abs(signal).max()
            if max_abs > 1e-8:
                signal = signal / max_abs
            return signal

        if t >= 5:
            mom = self.returns[t - 5:t].sum(axis=0)
            max_abs = np.abs(mom).max()
            if max_abs > 1e-8:
                mom = mom / max_abs
            return mom
        return None

class TiltPortfolioEnv(PortfolioEnv):

    def __init__(self, tilt_scale: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.tilt_scale = tilt_scale

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        equal = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        tilted = equal * (1.0 + self.tilt_scale * action)

        tilted = np.clip(tilted, 0.0, self.max_weight).astype(np.float32)

        total = tilted.sum()
        if total > 0:
            tilted = tilted / total
        else:
            tilted = equal

        return tilted
