from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward, BacktestResult
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import _BaseRLModel, _build_price_features
from src.rl.environment import TiltPortfolioEnv, PortfolioEnv

import warnings
warnings.filterwarnings("ignore", message=".*invalid value.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "concentrated"
MODELS_DIR = PROJECT_ROOT / "models"

class ConcentratedEnv(PortfolioEnv):

    def __init__(self, top_k: int = 20, ranking_window: int = 60, **kwargs):
        self.top_k = top_k
        self.ranking_window = ranking_window
        self._full_n_assets = kwargs.get("returns").shape[1]
        super().__init__(**kwargs)

        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(top_k,), dtype=np.float32
        )

    def _get_top_k_indices(self, t: int) -> np.ndarray:

        start = max(0, t - self.ranking_window)
        if start == t:
            return np.arange(self.top_k)
        window_returns = self.returns[start:t]
        cum_returns = (1 + window_returns).prod(axis=0) - 1
        cum_returns = np.nan_to_num(cum_returns, nan=-999)
        top_indices = np.argsort(cum_returns)[-self.top_k:]
        return top_indices

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        indices = self._get_top_k_indices(self._step)

        w_topk = np.clip(action, 0.0, None).astype(np.float32)
        w_topk = np.minimum(w_topk, self.max_weight * self.top_k)

        total = w_topk.sum()
        if total > 0:
            w_topk = w_topk / total
        else:
            w_topk = np.ones(self.top_k, dtype=np.float32) / self.top_k

        full_weights = np.zeros(self.n_assets, dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx < self.n_assets:
                full_weights[idx] = w_topk[i]

        full_weights = np.minimum(full_weights, self.max_weight)
        total = full_weights.sum()
        if total > 0:
            full_weights = full_weights / total
        return full_weights

    def _get_obs(self) -> np.ndarray:

        t = self._step
        indices = self._get_top_k_indices(t)

        parts = []

        if self.features_per_asset is not None:
            lb = self.lookback
            start = max(0, t - lb)
            for idx in indices:
                if idx < self.features_per_asset.shape[1]:
                    feat = self.features_per_asset[start:t, idx, :].flatten()
                else:
                    feat = np.zeros(lb * self.features_per_asset.shape[2])
                parts.append(feat)

        if self.features_global is not None:
            lb = self.lookback
            start = max(0, t - lb)
            parts.append(self.features_global[start:t].flatten())

        current_topk = np.array([self._weights[idx] if idx < len(self._weights) else 0
                                  for idx in indices])
        parts.append(current_topk)

        obs = np.concatenate(parts).astype(np.float32)

        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

class ConcentratedPPO(_BaseRLModel):

    def __init__(self, top_k: int = 20, timesteps: int = 1_000_000,
                 reward_type: str = "composite", device: str = "cpu", **kwargs):
        super().__init__(
            algorithm="PPO", total_timesteps=timesteps,
            reward_type=reward_type, use_vec_normalize=True,
            lookback=10, max_weight=0.15, seed=42,
            net_arch=[256, 128],
        )
        self.top_k = top_k
        self._device = device

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns, features_per_asset=features_pa,
            features_global=features_global, lookback=self.lookback,
            transaction_cost_bps=10.0, slippage_bps=5.0,
            reward_type=self.reward_type, max_weight=self.max_weight,
            top_k=self.top_k,
        )
        env = DummyVecEnv([lambda: ConcentratedEnv(**env_kwargs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                          clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return env

    def _create_agent(self, env, seed_val):
        pk = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
            batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            policy_kwargs=pk, verbose=0, seed=seed_val, device=self._device,
        )

class AggressiveTiltPPO(_BaseRLModel):

    def __init__(self, tilt_scale: float = 1.5, timesteps: int = 1_000_000,
                 reward_type: str = "composite", max_weight: float = 0.10,
                 device: str = "cpu", **kwargs):
        super().__init__(
            algorithm="PPO", total_timesteps=timesteps,
            reward_type=reward_type, use_vec_normalize=True,
            lookback=10, max_weight=max_weight, seed=42,
            net_arch=[256, 256],
        )
        self.tilt_scale = tilt_scale
        self._device = device

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns, features_per_asset=features_pa,
            features_global=features_global, lookback=self.lookback,
            transaction_cost_bps=10.0, slippage_bps=5.0,
            reward_type=self.reward_type, max_weight=self.max_weight,
            tilt_scale=self.tilt_scale,
        )
        env = DummyVecEnv([lambda: TiltPortfolioEnv(**env_kwargs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                          clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return env

    def _create_agent(self, env, seed_val):
        pk = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env, learning_rate=3.5e-4, n_steps=2048,
            batch_size=128, n_epochs=10, gamma=0.982, gae_lambda=0.976,
            clip_range=0.3, ent_coef=8e-4, vf_coef=0.47,
            policy_kwargs=pk, verbose=0, seed=seed_val, device=self._device,
        )

class MetaRLBlender:

    def __init__(self, timesteps: int = 500_000, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device
        self._agent = None
        self._vec_normalize = None

    def train(self, train_returns: np.ndarray,
              mvo_weights_history: np.ndarray,
              lstm_weights_history: np.ndarray):

        from gymnasium import Env, spaces

        T, N = train_returns.shape
        ew_weights = np.ones((T, N), dtype=np.float32) / N

        class MetaEnv(Env):

            def __init__(self_env):
                super().__init__()
                self_env.observation_space = spaces.Box(-10, 10, (12,), np.float32)
                self_env.action_space = spaces.Box(0, 1, (3,), np.float32)
                self_env._step = 60
                self_env._portfolio_returns = []

            def reset(self_env, seed=None, options=None):
                super().reset(seed=seed)
                self_env._step = 60
                self_env._portfolio_returns = []
                return self_env._get_obs(), {}

            def _get_obs(self_env):
                t = self_env._step
                r = train_returns[max(0, t-60):t]
                mkt = r.mean(axis=1)
                obs = np.zeros(12, dtype=np.float32)
                if len(mkt) >= 20:
                    obs[0] = np.std(mkt[-20:]) * np.sqrt(252)
                    obs[1] = np.std(mkt[-60:]) * np.sqrt(252)
                    obs[2] = np.mean(mkt[-5:]) * 252
                    obs[3] = np.mean(mkt[-20:]) * 252
                    obs[4] = np.mean(mkt[-60:]) * 252
                    pos_frac = (r[-20:].mean(axis=0) > 0).mean()
                    obs[5] = pos_frac
                    obs[6] = np.std(r[-20:].mean(axis=0))

                    for i, wh in enumerate([mvo_weights_history, lstm_weights_history, ew_weights]):
                        if t >= 20:
                            rets = np.sum(wh[t-20:t] * train_returns[t-20:t], axis=1)
                            obs[7+i] = np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(252)
                    obs[10] = t / T
                    obs[11] = np.mean(mkt)
                return np.nan_to_num(obs, nan=0.0).astype(np.float32)

            def step(self_env, action):
                t = self_env._step

                w = np.clip(action, 0.01, None)
                w = w / w.sum()

                blended = (w[0] * mvo_weights_history[t] +
                          w[1] * lstm_weights_history[t] +
                          w[2] * ew_weights[t])
                blended = np.clip(blended, 0, None)
                s = blended.sum()
                if s > 0:
                    blended = blended / s

                port_ret = np.sum(blended * train_returns[t])

                if t > 60:
                    prev_blended = (w[0] * mvo_weights_history[t-1] +
                                   w[1] * lstm_weights_history[t-1] +
                                   w[2] * ew_weights[t-1])
                    prev_s = prev_blended.sum()
                    if prev_s > 0:
                        prev_blended = prev_blended / prev_s
                    tc = np.abs(blended - prev_blended).sum() * 0.001
                    port_ret -= tc

                self_env._portfolio_returns.append(port_ret)

                rets = np.array(self_env._portfolio_returns[-60:])
                if len(rets) > 5:
                    reward = np.mean(rets) / max(np.std(rets), 1e-6) * 0.1
                else:
                    reward = port_ret * 10

                self_env._step += 1
                done = self_env._step >= T
                obs = self_env._get_obs() if not done else np.zeros(12, np.float32)
                return obs, float(reward), done, False, {"port_return": port_ret, "blend": w.copy()}

        env = DummyVecEnv([MetaEnv])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

        agent = PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=1024,
            batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.02,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0, seed=42, device=self.device,
        )
        agent.learn(total_timesteps=self.timesteps)
        self._agent = agent
        self._vec_normalize = env

    def predict_blend(self, obs: np.ndarray) -> np.ndarray:

        action, _ = self._agent.predict(obs, deterministic=True)
        w = np.clip(action, 0.01, None)
        return w / w.sum()

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
    logger.info(
        "%s: %.0fs | Sharpe=%.3f | Ret=%.2f%% | DD=%.2f%% | Calmar=%.3f | Sortino=%.3f",
        name, elapsed, metrics["Sharpe Ratio"],
        metrics["Cumulative Return"] * 100, metrics["Max Drawdown"] * 100,
        metrics["Calmar Ratio"], metrics["Sortino Ratio"],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace(" ", "_")
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)
    return result, metrics

def main():
    from src.rl.train_full_pipeline import FullPipelineTiltPPO

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets", train_split.n_days, n)

    all_results = {}

    logger.info("\n" + "▸" * 30 + " PHASE 1: AGGRESSIVE TILT " + "◂" * 30)
    for ts in [0.8, 1.5, 3.0]:
        name = f"AggressiveTilt_ts{ts:.1f}_1M"
        model = AggressiveTiltPPO(
            tilt_scale=ts, timesteps=1_000_000,
            reward_type="composite", max_weight=0.10, device="cpu",
        )
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    logger.info("\n" + "▸" * 30 + " PHASE 2: CONCENTRATED TOP-K " + "◂" * 30)
    for k in [15, 25]:
        name = f"Concentrated_top{k}_1M"
        model = ConcentratedPPO(
            top_k=k, timesteps=1_000_000,
            reward_type="composite", device="cpu",
        )
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    logger.info("\n" + "▸" * 30 + " PHASE 3: DIRECT WEIGHT PPO " + "◂" * 30)
    from src.rl.agent import PPOModel
    for mw in [0.05, 0.10]:
        name = f"DirectPPO_mw{int(mw*100):02d}_1M"
        model = PPOModel(
            total_timesteps=1_000_000, reward_type="composite",
            use_vec_normalize=True, lookback=10, max_weight=mw,
            seed=42, net_arch=[256, 256],
        )
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    print("\n" + "=" * 80)
    print("CONCENTRATED/AGGRESSIVE RL RESULTS")
    print("=" * 80)
    print(f"  {'Model':<35s} | {'Sharpe':>7s} | {'Return':>8s} | {'MaxDD':>7s} | {'Calmar':>7s}")
    print("-" * 80)
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True):
        print(f"  {name:<35s} | {m['Sharpe Ratio']:7.3f} | {m['Cumulative Return']*100:7.2f}% | "
              f"{m['Max Drawdown']*100:6.2f}% | {m['Calmar Ratio']:7.3f}")
    print("=" * 80)

    summary = pd.DataFrame(all_results).T
    summary.to_csv(RESULTS_DIR / "concentrated_summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "concentrated_summary.csv")

if __name__ == "__main__":
    main()
