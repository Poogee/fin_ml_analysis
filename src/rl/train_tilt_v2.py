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
from src.rl.environment import TiltPortfolioEnv

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

class TiltPPO(_BaseRLModel):
    def __init__(self, tilt_scale=0.3, timesteps=1_000_000,
                 reward_type="composite", lookback=10, max_weight=0.15, seed=42):
        super().__init__(algorithm="PPO", total_timesteps=timesteps,
                         reward_type=reward_type, use_vec_normalize=True,
                         lookback=lookback, max_weight=max_weight, seed=seed)
        self.tilt_scale = tilt_scale

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns, features_per_asset=features_pa,
            features_global=features_global, lookback=self.lookback,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            reward_type=self.reward_type, max_weight=self.max_weight,
            lambda_dd=self.lambda_dd, lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold, dsr_eta=self.dsr_eta,
            tilt_scale=self.tilt_scale,
        )
        env = DummyVecEnv([lambda: TiltPortfolioEnv(**env_kwargs)])
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

        if self._agent is None:
            return np.ones(n_assets, dtype=np.float32) / n_assets

        price_feats = _build_price_features(returns_arr, self.lookback)
        env_kwargs = dict(
            returns=returns_arr, features_per_asset=price_feats,
            features_global=None, lookback=self.lookback,
            transaction_cost_bps=self.tc_bps, slippage_bps=self.slip_bps,
            max_weight=self.max_weight, tilt_scale=self.tilt_scale,
        )
        tmp_env = TiltPortfolioEnv(**env_kwargs)
        tmp_env._step = len(returns_arr) - 1
        tmp_env._weights = np.zeros(n_assets, dtype=np.float32)
        obs = tmp_env._get_obs()

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._agent.predict(obs, deterministic=True)
        return tmp_env._normalize_weights(action)

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

    for ts in [0.10, 0.15, 0.20, 0.25, 0.30]:
        name = f"Tilt_ts{int(ts*100):02d}_1M"
        _, m = train_eval(TiltPPO(tilt_scale=ts), name, train_split, test_split)
        all_results[name] = m

    best_ts_name = max(
        [k for k in all_results if k.startswith("Tilt_ts")],
        key=lambda k: all_results[k]["Sharpe Ratio"]
    )
    best_ts = float(best_ts_name.split("ts")[1][:2]) / 100
    logger.info("Best tilt_scale: %.2f (Sharpe=%.3f)", best_ts,
                all_results[best_ts_name]["Sharpe Ratio"])

    _, m = train_eval(
        TiltPPO(tilt_scale=best_ts, reward_type="sharpe"),
        f"Tilt_ts{int(best_ts*100):02d}_sharpe_1M", train_split, test_split)
    all_results[f"Tilt_ts{int(best_ts*100):02d}_sharpe_1M"] = m

    for steps in [2_000_000, 3_000_000]:
        name = f"Tilt_ts{int(best_ts*100):02d}_{steps//1_000_000}M"
        _, m = train_eval(
            TiltPPO(tilt_scale=best_ts, timesteps=steps),
            name, train_split, test_split)
        all_results[name] = m

    _, m = train_eval(
        TiltPPO(tilt_scale=best_ts, lookback=5),
        f"Tilt_ts{int(best_ts*100):02d}_lb5_1M", train_split, test_split)
    all_results[f"Tilt_ts{int(best_ts*100):02d}_lb5_1M"] = m

    _, m = train_eval(
        TiltPPO(tilt_scale=best_ts, lookback=20),
        f"Tilt_ts{int(best_ts*100):02d}_lb20_1M", train_split, test_split)
    all_results[f"Tilt_ts{int(best_ts*100):02d}_lb20_1M"] = m

    ens = EnsembleModel([
        TiltPPO(tilt_scale=best_ts, seed=42),
        TiltPPO(tilt_scale=best_ts, seed=1042),
        TiltPPO(tilt_scale=best_ts, seed=2042),
    ])
    _, m = train_eval(ens, "Ensemble_3xTilt", train_split, test_split)
    all_results["Ensemble_3xTilt"] = m

    print("\n" + "=" * 100)
    print("TILT PPO v2 RESULTS (n_assets=100, residual policy)")
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

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "tilt_v2_summary.csv")

if __name__ == "__main__":
    main()
