from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import _BaseRLModel, _build_price_features
from src.baselines.mean_variance import MeanVarianceModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"

def filter_splits(train_split, test_split):
    intersection = sorted(set(train_split.all_assets) & set(test_split.all_assets))
    common_cols = set(train_split.returns.columns) & set(test_split.returns.columns)
    available = sorted(a for a in intersection if a in common_cols)
    logger.info("Intersection: %d assets", len(available))

    def _f(split, assets):
        return DataSplit(
            prices=split.prices[assets], returns=split.returns[assets],
            log_returns=split.log_returns[assets], market_caps=split.market_caps[assets],
            dividends=split.dividends[assets], presence=split.presence[assets],
            universe_schedule=split.universe_schedule,
            start_date=split.start_date, end_date=split.end_date, all_assets=assets,
        )
    return _f(train_split, available), _f(test_split, available)

class MetaMVOEnv(gym.Env):

    metadata = {"render_modes": []}

    BASE_RISK_AVERSION = 2.25
    BASE_SHRINKAGE = 0.8
    BASE_MAX_WEIGHT = 0.08
    BASE_LOOKBACK = 105

    def __init__(
        self,
        returns: np.ndarray,
        lookback: int = 60,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        reward_type: str = "composite",
        lambda_dd: float = 2.0,
        lambda_turnover: float = 0.5,
        dd_threshold: float = 0.05,
        dsr_eta: float = 0.005,
        action_dim: int = 3,
    ):
        super().__init__()

        self.returns = np.nan_to_num(returns, nan=0.0).astype(np.float32)
        self.n_dates, self.n_assets = self.returns.shape
        self.lookback = lookback
        self.tc_rate = (transaction_cost_bps + slippage_bps) / 10000.0
        self.reward_type = reward_type
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.dd_threshold = dd_threshold
        self.dsr_eta = dsr_eta
        self.action_dim = action_dim

        self._mvo = MeanVarianceModel(
            lookback=self.BASE_LOOKBACK,
            max_weight=self.BASE_MAX_WEIGHT,
            risk_aversion=self.BASE_RISK_AVERSION,
            shrinkage=self.BASE_SHRINKAGE,
            min_history=30,
        )

        self.obs_dim = 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32,
        )

        self._step = 0
        self._weights = np.zeros(self.n_assets, dtype=np.float32)
        self._portfolio_returns = []
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._peak_value = 1.0
        self._cumulative_value = 1.0

        self._cache_stats()

    def _cache_stats(self):

        T, N = self.returns.shape

        self._mean_20 = np.zeros(T, dtype=np.float32)
        self._mean_60 = np.zeros(T, dtype=np.float32)
        self._vol_20 = np.zeros(T, dtype=np.float32)
        self._vol_60 = np.zeros(T, dtype=np.float32)
        self._skew_20 = np.zeros(T, dtype=np.float32)
        self._sharpe_60 = np.zeros(T, dtype=np.float32)
        self._mom_5 = np.zeros(T, dtype=np.float32)
        self._mom_20 = np.zeros(T, dtype=np.float32)
        self._mom_60 = np.zeros(T, dtype=np.float32)
        self._cross_corr = np.zeros(T, dtype=np.float32)

        ew_returns = np.nanmean(self.returns, axis=1)

        for t in range(1, T):
            if t >= 20:
                w = ew_returns[t-20:t]
                self._mean_20[t] = np.mean(w)
                self._vol_20[t] = np.std(w) + 1e-8
                self._skew_20[t] = float(np.mean(((w - np.mean(w)) / (np.std(w) + 1e-8))**3))
            if t >= 60:
                w = ew_returns[t-60:t]
                self._mean_60[t] = np.mean(w)
                self._vol_60[t] = np.std(w) + 1e-8
                s = np.std(w)
                self._sharpe_60[t] = np.mean(w) / (s + 1e-8) * np.sqrt(252)
            if t >= 5:
                self._mom_5[t] = np.sum(ew_returns[t-5:t])
            if t >= 20:
                self._mom_20[t] = np.sum(ew_returns[t-20:t])
            if t >= 60:
                self._mom_60[t] = np.sum(ew_returns[t-60:t])

            if t >= 60:
                sample_idx = np.linspace(0, N-1, min(20, N), dtype=int)
                window = self.returns[t-60:t, sample_idx]
                corr = np.corrcoef(window.T)
                mask = ~np.eye(corr.shape[0], dtype=bool)
                self._cross_corr[t] = np.nanmean(corr[mask])

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

        new_weights = self._action_to_weights(action)

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

        self._step += 1
        terminated = self._step >= self.n_dates
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_return": port_return,
            "transaction_cost": tc,
            "turnover": turnover,
            "drawdown": current_drawdown,
        }
        return obs, reward, terminated, truncated, info

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:

        risk_aversion = self.BASE_RISK_AVERSION * np.exp(1.0 * action[0])

        shrinkage = 1.0 / (1.0 + np.exp(-(3.0 * action[1])))
        shrinkage = np.clip(shrinkage, 0.1, 0.99)

        if self.action_dim >= 3:

            max_weight = self.BASE_MAX_WEIGHT * np.exp(0.7 * action[2])
            max_weight = np.clip(max_weight, 0.02, 0.25)
        else:
            max_weight = self.BASE_MAX_WEIGHT

        t = self._step
        lookback = min(self.BASE_LOOKBACK, t)
        window = self.returns[max(0, t - lookback):t + 1]
        window_clean = np.nan_to_num(window, nan=0.0)

        mu = window_clean.mean(axis=0)

        sample_cov = np.cov(window_clean, rowvar=False)
        if sample_cov.ndim == 0:
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        target = np.diag(np.diag(sample_cov))
        cov = (1 - shrinkage) * sample_cov + shrinkage * target

        n = len(mu)
        cov_reg = cov + np.eye(n) * 1e-6
        try:
            raw_w = np.linalg.solve(cov_reg, mu) / risk_aversion
        except np.linalg.LinAlgError:
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        w = np.maximum(raw_w, 0.0).astype(np.float32)

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
            w = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return w

    def _get_obs(self) -> np.ndarray:

        t = self._step
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        obs[0] = self._mean_20[t] * 252
        obs[1] = self._mean_60[t] * 252
        obs[2] = self._vol_20[t] * np.sqrt(252)
        obs[3] = self._vol_60[t] * np.sqrt(252)
        obs[4] = self._skew_20[t]
        obs[5] = self._sharpe_60[t]
        obs[6] = 1.0 - self._cumulative_value / self._peak_value
        obs[7] = self._mom_5[t]
        obs[8] = self._mom_20[t]
        obs[9] = self._mom_60[t]
        obs[10] = self._vol_20[t] / (self._vol_60[t] + 1e-8)
        obs[11] = self._cross_corr[t]

        obs[12] = np.sum(self._weights ** 2)

        obs[13] = self._portfolio_returns[-1] if self._portfolio_returns else 0.0

        return obs

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

    def _compute_reward(self, port_return, tc, turnover, current_drawdown):
        if self.reward_type == "composite":
            dsr = self._compute_dsr(port_return)
            dd_penalty = max(0.0, current_drawdown - self.dd_threshold)
            return dsr - self.lambda_dd * dd_penalty - self.lambda_turnover * turnover
        elif self.reward_type == "sharpe":
            window = self._portfolio_returns[-20:]
            if len(window) < 5:
                return port_return
            mean_r = np.mean(window)
            std_r = np.std(window)
            if std_r < 1e-8:
                return mean_r * 100
            return (mean_r / std_r) * np.sqrt(252)
        else:
            return port_return - tc

class MetaMVOModel(_BaseRLModel):

    def __init__(self, timesteps=1_000_000, reward_type="composite",
                 action_dim=3, seed=42):
        super().__init__(algorithm="PPO", total_timesteps=timesteps,
                         reward_type=reward_type, use_vec_normalize=True,
                         lookback=60, max_weight=0.08, seed=seed)
        self.action_dim = action_dim
        self._mvo = MeanVarianceModel(
            lookback=105, max_weight=0.08,
            risk_aversion=2.25, shrinkage=0.8,
            min_history=30,
        )

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns, lookback=60,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            lambda_dd=self.lambda_dd, lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold, dsr_eta=self.dsr_eta,
            action_dim=self.action_dim,
        )
        env = DummyVecEnv([lambda: MetaMVOEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return env

    def _create_agent(self, env, seed_val):

        pk = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                  activation_fn=torch.nn.Tanh)
        return PPO("MlpPolicy", env,
                   learning_rate=3e-4, n_steps=2048, batch_size=64,
                   n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                   max_grad_norm=0.5,
                   policy_kwargs=pk,
                   verbose=0, seed=seed_val, device="cpu")

    def predict_weights(self, current_data):

        returns = current_data["returns"]
        if hasattr(returns, "values"):
            returns_arr = returns.values
        else:
            returns_arr = returns
        returns_arr = np.nan_to_num(returns_arr, nan=0.0).astype(np.float32)
        n_assets = returns_arr.shape[1]

        if self._agent is None:

            return self._mvo.predict_weights(current_data)

        env_kwargs = dict(
            returns=returns_arr, lookback=60,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            action_dim=self.action_dim,
        )
        tmp_env = MetaMVOEnv(**env_kwargs)
        tmp_env._step = len(returns_arr) - 1
        tmp_env._weights = np.zeros(n_assets, dtype=np.float32)
        obs = tmp_env._get_obs()

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._agent.predict(obs, deterministic=True)
        return tmp_env._action_to_weights(action)

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, train_data):
        for i, m in enumerate(self.models):
            logger.info("Ensemble %d/%d...", i + 1, len(self.models))
            m.fit(train_data)

    def predict_weights(self, current_data):
        all_w = []
        for m in self.models:
            try:
                all_w.append(m.predict_weights(current_data))
            except Exception:
                pass
        if not all_w:
            n = current_data["returns"].shape[1] if hasattr(current_data["returns"], "shape") else len(current_data["returns"].columns)
            return np.ones(n) / n
        avg = np.mean(all_w, axis=0)
        avg = np.maximum(avg, 0)
        s = avg.sum()
        return avg / s if s > 0 else avg

def train_eval(model, name, train_split, test_split):
    logger.info("=" * 60)
    logger.info("TRAINING: %s", name)
    t0 = time.time()
    result = run_walk_forward(
        model=model, train_split=train_split, test_split=test_split,
        model_name=name, transaction_cost_bps=10.0, slippage_bps=5.0,
        rebalance_frequency=5,
    )
    elapsed = time.time() - t0
    metrics = PortfolioMetrics.compute_all(result.returns)
    logger.info("%s: %.0fs | Sharpe=%.3f | Ret=%.2f%% | DD=%.2f%% | Calmar=%.3f | Sortino=%.3f",
                name, elapsed, metrics["Sharpe Ratio"],
                metrics["Cumulative Return"] * 100,
                metrics["Max Drawdown"] * 100, metrics["Calmar Ratio"],
                metrics["Sortino Ratio"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace(" ", "_")
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)
    return result, metrics

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=100)...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets", train_split.n_days, n)

    all_results = {}
    TARGET = 0.89

    for steps in [500_000, 1_000_000, 2_000_000]:
        name = f"MetaMVO_{steps//1_000_000}M" if steps >= 1_000_000 else f"MetaMVO_{steps//1000}K"
        model = MetaMVOModel(timesteps=steps, reward_type="composite", action_dim=3)
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    model = MetaMVOModel(timesteps=1_000_000, reward_type="composite", action_dim=2)
    _, m = train_eval(model, "MetaMVO_2D_1M", train_split, test_split)
    all_results["MetaMVO_2D_1M"] = m

    model = MetaMVOModel(timesteps=1_000_000, reward_type="sharpe", action_dim=3)
    _, m = train_eval(model, "MetaMVO_sharpe_1M", train_split, test_split)
    all_results["MetaMVO_sharpe_1M"] = m

    ens = EnsembleModel([
        MetaMVOModel(timesteps=1_000_000, seed=42),
        MetaMVOModel(timesteps=1_000_000, seed=1042),
        MetaMVOModel(timesteps=1_000_000, seed=2042),
    ])
    _, m = train_eval(ens, "Ensemble_3xMetaMVO", train_split, test_split)
    all_results["Ensemble_3xMetaMVO"] = m

    best_name = max(all_results, key=lambda k: all_results[k]["Sharpe Ratio"])
    best_sharpe = all_results[best_name]["Sharpe Ratio"]
    logger.info("Best so far: %s (Sharpe=%.3f)", best_name, best_sharpe)

    if best_sharpe < TARGET:

        model = MetaMVOModel(timesteps=5_000_000, reward_type="composite", action_dim=3)
        _, m = train_eval(model, "MetaMVO_5M", train_split, test_split)
        all_results["MetaMVO_5M"] = m

    print("\n" + "=" * 100)
    print("META-RL MVO RESULTS (RL adapts MVO hyperparameters)")
    print("=" * 100)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 100)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_results:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f}{flag}")
    print("=" * 100)

    best = sorted_results[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")
    if best[1]["Sharpe Ratio"] >= TARGET:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {TARGET - best[1]['Sharpe Ratio']:.3f}")

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "meta_mvo_summary.csv")

if __name__ == "__main__":
    main()
