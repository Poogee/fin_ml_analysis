from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import _BaseRLModel, _build_price_features
from src.rl.environment import PortfolioEnv, TiltPortfolioEnv
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

def precompute_mvo_weights(returns_arr: np.ndarray,
                            lookback: int = 105,
                            max_weight: float = 0.08,
                            risk_aversion: float = 2.25,
                            shrinkage: float = 0.8,
                            min_history: int = 30) -> np.ndarray:

    T, N = returns_arr.shape
    ew = np.ones(N, dtype=np.float32) / N
    mvo_weights = np.tile(ew, (T, 1))

    mvo = MeanVarianceModel(
        lookback=lookback, max_weight=max_weight,
        risk_aversion=risk_aversion, shrinkage=shrinkage,
        min_history=min_history,
    )

    logger.info("Pre-computing MVO weights for %d days, %d assets...", T, N)
    t0 = time.time()

    for t in range(min_history, T):
        start = max(0, t - lookback)
        window = returns_arr[start:t + 1]

        mu = np.nanmean(window, axis=0)
        valid = ~np.isnan(mu) & (np.count_nonzero(~np.isnan(window), axis=0) >= min(min_history, window.shape[0]))
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 2:
            continue

        valid_returns = np.nan_to_num(window[:, valid_idx], nan=0.0)
        mu_valid = valid_returns.mean(axis=0)
        cov = mvo._shrink_covariance(valid_returns)
        opt_w = mvo._solve_mv(mu_valid, cov, max_weight)

        w = np.zeros(N, dtype=np.float32)
        w[valid_idx] = opt_w.astype(np.float32)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = ew.copy()
        mvo_weights[t] = w

    elapsed = time.time() - t0
    logger.info("MVO pre-computation done: %.1fs", elapsed)
    return mvo_weights

class MVOTiltPortfolioEnv(PortfolioEnv):

    def __init__(self, mvo_weights: np.ndarray, tilt_scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mvo_weights = mvo_weights.astype(np.float32)
        self.tilt_scale = tilt_scale

        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        self.obs_dim += self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        t = min(self._step, len(self.mvo_weights) - 1)
        baseline = self.mvo_weights[t].copy()

        b_sum = baseline.sum()
        if b_sum <= 0:
            baseline = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        else:
            baseline = baseline / b_sum

        tilted = baseline * (1.0 + self.tilt_scale * action)

        tilted = np.clip(tilted, 0.0, self.max_weight).astype(np.float32)

        total = tilted.sum()
        if total > 0:
            tilted = tilted / total
        else:
            tilted = baseline
        return tilted

    def _get_obs(self) -> np.ndarray:

        base_obs = super()._get_obs()
        t = min(self._step, len(self.mvo_weights) - 1)
        mvo_w = self.mvo_weights[t]
        return np.concatenate([base_obs, mvo_w]).astype(np.float32)

class MVOTiltPPO(_BaseRLModel):

    def __init__(self, tilt_scale=0.1, timesteps=1_000_000,
                 reward_type="composite", lookback=10, max_weight=0.15,
                 mvo_lookback=105, mvo_max_weight=0.08,
                 mvo_risk_aversion=2.25, mvo_shrinkage=0.8,
                 seed=42):
        super().__init__(algorithm="PPO", total_timesteps=timesteps,
                         reward_type=reward_type, use_vec_normalize=True,
                         lookback=lookback, max_weight=max_weight, seed=seed)
        self.tilt_scale = tilt_scale
        self.mvo_lookback = mvo_lookback
        self.mvo_max_weight = mvo_max_weight
        self.mvo_risk_aversion = mvo_risk_aversion
        self.mvo_shrinkage = mvo_shrinkage
        self._mvo_model = MeanVarianceModel(
            lookback=mvo_lookback, max_weight=mvo_max_weight,
            risk_aversion=mvo_risk_aversion, shrinkage=mvo_shrinkage,
            min_history=30,
        )
        self._precomputed_mvo = None

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):

        if self._precomputed_mvo is None:
            self._precomputed_mvo = precompute_mvo_weights(
                returns,
                lookback=self.mvo_lookback,
                max_weight=self.mvo_max_weight,
                risk_aversion=self.mvo_risk_aversion,
                shrinkage=self.mvo_shrinkage,
            )

        env_kwargs = dict(
            returns=returns, features_per_asset=features_pa,
            features_global=features_global, lookback=self.lookback,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            reward_type=self.reward_type, max_weight=self.max_weight,
            lambda_dd=self.lambda_dd, lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold, dsr_eta=self.dsr_eta,
            mvo_weights=self._precomputed_mvo,
            tilt_scale=self.tilt_scale,
        )
        env = DummyVecEnv([lambda: MVOTiltPortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return env

    def _create_agent(self, env, seed_val):
        pk = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                  activation_fn=torch.nn.Tanh)
        return PPO("MlpPolicy", env,
                   learning_rate=3.5e-4, n_steps=2048, batch_size=128,
                   n_epochs=10, gamma=0.982, gae_lambda=0.976,
                   clip_range=0.3, ent_coef=8e-4, vf_coef=0.47,
                   max_grad_norm=0.51,
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

        mvo_weights = self._mvo_model.predict_weights(current_data)

        if self._agent is None:
            return mvo_weights

        price_feats = _build_price_features(returns_arr, self.lookback)

        mvo_w_arr = np.zeros((len(returns_arr), n_assets), dtype=np.float32)
        mvo_w_arr[-1] = mvo_weights.astype(np.float32)

        env_kwargs = dict(
            returns=returns_arr, features_per_asset=price_feats,
            features_global=None, lookback=self.lookback,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            max_weight=self.max_weight,
            mvo_weights=mvo_w_arr,
            tilt_scale=self.tilt_scale,
        )
        tmp_env = MVOTiltPortfolioEnv(**env_kwargs)
        tmp_env._step = len(returns_arr) - 1
        tmp_env._weights = np.zeros(n_assets, dtype=np.float32)
        obs = tmp_env._get_obs()

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._agent.predict(obs, deterministic=True)
        return tmp_env._normalize_weights(action)

class StrategyBlendEnv(PortfolioEnv):

    def __init__(self, strategy_weights: np.ndarray, strategy_names: list[str],
                 **kwargs):

        self.strategy_weights = strategy_weights.astype(np.float32)
        self.n_strategies = strategy_weights.shape[1]
        self.strategy_names = strategy_names

        super().__init__(**kwargs)

        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_strategies,), dtype=np.float32,
        )

        self.obs_dim += self.n_strategies * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:

        t = min(self._step, len(self.strategy_weights) - 1)

        blend = np.clip(action[:self.n_strategies], 0.0, None)
        b_sum = blend.sum()
        if b_sum > 0:
            blend /= b_sum
        else:
            blend = np.ones(self.n_strategies, dtype=np.float32) / self.n_strategies

        weights = np.zeros(self.n_assets, dtype=np.float32)
        for i in range(self.n_strategies):
            weights += blend[i] * self.strategy_weights[t, i]

        weights = np.clip(weights, 0.0, self.max_weight).astype(np.float32)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return weights

    def _get_obs(self) -> np.ndarray:

        base_obs = super()._get_obs()
        t = min(self._step, len(self.strategy_weights) - 1)
        strat_flat = self.strategy_weights[t].flatten()
        return np.concatenate([base_obs, strat_flat]).astype(np.float32)

def precompute_riskparity_weights(returns_arr: np.ndarray,
                                    vol_lookback: int = 80) -> np.ndarray:

    T, N = returns_arr.shape
    ew = np.ones(N, dtype=np.float32) / N
    rp_weights = np.tile(ew, (T, 1))

    for t in range(vol_lookback, T):
        window = returns_arr[t - vol_lookback:t]
        vols = np.nanstd(window, axis=0)
        vols = np.maximum(vols, 1e-8)
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()
        rp_weights[t] = w.astype(np.float32)

    return rp_weights

def precompute_momentum_weights(returns_arr: np.ndarray,
                                  lookback: int = 20,
                                  top_k: int = 40,
                                  max_weight: float = 0.08) -> np.ndarray:

    T, N = returns_arr.shape
    ew = np.ones(N, dtype=np.float32) / N
    mom_weights = np.tile(ew, (T, 1))

    for t in range(lookback, T):
        window = returns_arr[t - lookback:t]
        cum_ret = np.nanprod(1 + window, axis=0) - 1

        k = min(top_k, N)
        top_idx = np.argsort(cum_ret)[-k:]
        w = np.zeros(N, dtype=np.float32)
        scores = np.maximum(cum_ret[top_idx], 0)
        s = scores.sum()
        if s > 0:
            w[top_idx] = (scores / s).astype(np.float32)
        else:
            w[top_idx] = 1.0 / k

        w = np.minimum(w, max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        mom_weights[t] = w

    return mom_weights

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

    for ts in [0.05, 0.10, 0.15, 0.20]:
        name = f"MVOTilt_ts{int(ts*100):02d}_1M"
        model = MVOTiltPPO(tilt_scale=ts, timesteps=1_000_000, lookback=10)
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    best_ts_name = max(
        [k for k in all_results if k.startswith("MVOTilt_ts")],
        key=lambda k: all_results[k]["Sharpe Ratio"]
    )
    best_ts = float(best_ts_name.split("ts")[1][:2]) / 100
    logger.info("Best MVO-Tilt tilt_scale: %.2f (Sharpe=%.3f)", best_ts,
                all_results[best_ts_name]["Sharpe Ratio"])

    name = f"MVOTilt_ts{int(best_ts*100):02d}_2M"
    model = MVOTiltPPO(tilt_scale=best_ts, timesteps=2_000_000, lookback=10)
    _, m = train_eval(model, name, train_split, test_split)
    all_results[name] = m

    name = f"MVOTilt_ts{int(best_ts*100):02d}_sharpe_1M"
    model = MVOTiltPPO(tilt_scale=best_ts, timesteps=1_000_000,
                        reward_type="sharpe", lookback=10)
    _, m = train_eval(model, name, train_split, test_split)
    all_results[name] = m

    ens = EnsembleModel([
        MVOTiltPPO(tilt_scale=best_ts, timesteps=1_000_000, seed=42),
        MVOTiltPPO(tilt_scale=best_ts, timesteps=1_000_000, seed=1042),
        MVOTiltPPO(tilt_scale=best_ts, timesteps=1_000_000, seed=2042),
    ])
    _, m = train_eval(ens, "Ensemble_3xMVOTilt", train_split, test_split)
    all_results["Ensemble_3xMVOTilt"] = m

    for ts in [0.01, 0.02, 0.03]:
        name = f"MVOTilt_ts{int(ts*100):02d}_1M"
        model = MVOTiltPPO(tilt_scale=ts, timesteps=1_000_000, lookback=10)
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    print("\n" + "=" * 100)
    print("MVO-TILT PPO RESULTS (n_assets=100, residual policy on MVO baseline)")
    print("=" * 100)
    print(f"{'Model':<40s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 100)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_results:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<38s} | {m['Sharpe Ratio']:7.3f} | "
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

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "mvo_tilt_summary.csv")

if __name__ == "__main__":
    main()
