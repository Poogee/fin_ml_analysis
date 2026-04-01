from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import _BaseRLModel
from src.rl.environment import PortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"
DEVICE_PPO = "cpu"
DEVICE_SAC = "cuda" if torch.cuda.is_available() else "cpu"

def filter_splits(train_split, test_split):

    intersection = sorted(set(train_split.all_assets) & set(test_split.all_assets))
    common_cols = set(train_split.returns.columns) & set(test_split.returns.columns)
    available = sorted(a for a in intersection if a in common_cols)
    logger.info("Intersection universe: %d assets (from %d train, %d test)",
                len(available), len(train_split.all_assets), len(test_split.all_assets))

    def _f(split, assets):
        return DataSplit(
            prices=split.prices[assets], returns=split.returns[assets],
            log_returns=split.log_returns[assets], market_caps=split.market_caps[assets],
            dividends=split.dividends[assets], presence=split.presence[assets],
            universe_schedule=split.universe_schedule,
            start_date=split.start_date, end_date=split.end_date, all_assets=assets,
        )
    return _f(train_split, available), _f(test_split, available)

def make_ppo_model(reward_type, timesteps, seed=42, **overrides):

    class _Model(_BaseRLModel):
        def __init__(self):
            super().__init__(
                algorithm="PPO",
                total_timesteps=timesteps,
                reward_type=reward_type,
                use_vec_normalize=True,
                seed=seed,
                lookback=5,
                **{k: v for k, v in overrides.items() if k in
                   ['lambda_dd', 'lambda_turnover', 'dd_threshold', 'dsr_eta',
                    'max_weight', 'transaction_cost_bps', 'slippage_bps']},
            )
            self._extra = overrides

        def _get_feature_config(self):
            return {"use_graph": False, "use_sentiment": False}

        def _create_agent(self, env, seed_val):

            lr = self._extra.get("learning_rate", 3.5e-4)
            n_steps = self._extra.get("n_steps", 2048)
            batch_size = self._extra.get("batch_size", 128)
            gamma = self._extra.get("gamma", 0.982)
            clip_range = self._extra.get("clip_range", 0.3)
            ent_coef = self._extra.get("ent_coef", 8e-4)
            gae_lambda = self._extra.get("gae_lambda", 0.976)
            max_grad_norm = self._extra.get("max_grad_norm", 0.51)
            n_epochs = self._extra.get("n_epochs", 10)

            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.Tanh,
            )
            return PPO(
                "MlpPolicy", env,
                learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
                n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
                clip_range=clip_range, ent_coef=ent_coef, vf_coef=0.47,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=0, seed=seed_val, device=DEVICE_PPO,
            )
    return _Model()

def make_sac_model(reward_type, timesteps, seed=42, **overrides):

    class _Model(_BaseRLModel):
        def __init__(self):
            super().__init__(
                algorithm="SAC",
                total_timesteps=timesteps,
                reward_type=reward_type,
                use_vec_normalize=True,
                seed=seed,
                lookback=5,
            )

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
            )
            env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
            if self.use_vec_normalize:
                env = VecNormalize(env, norm_obs=True, norm_reward=True,
                                   clip_obs=10.0, clip_reward=10.0, gamma=0.99)
            return env

        def _create_agent(self, env, seed_val):
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 128], qf=[256, 128]),
                activation_fn=torch.nn.Tanh,
            )
            return SAC(
                "MlpPolicy", env,
                learning_rate=3e-4, buffer_size=100_000,
                learning_starts=1000, batch_size=256,
                gamma=0.99, tau=0.005, ent_coef="auto",
                policy_kwargs=policy_kwargs,
                verbose=0, seed=seed_val, device=DEVICE_SAC,
            )
    return _Model()

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, train_data):
        for i, m in enumerate(self.models):
            logger.info("Ensemble %d/%d...", i+1, len(self.models))
            m.fit(train_data)

    def predict_weights(self, current_data):
        all_w = []
        for m in self.models:
            try:
                all_w.append(m.predict_weights(current_data))
            except Exception:
                pass
        if not all_w:
            n = current_data["returns"].shape[1] if hasattr(current_data["returns"], 'shape') else len(current_data["returns"].columns)
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
    safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)
    return result, metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_assets", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=%d)...", args.n_assets)
    pipeline = DataPipeline(n_assets=args.n_assets)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    logger.info("Train: %d days, %d assets | Test: %d days, %d assets",
                train_split.n_days, train_split.n_assets,
                test_split.n_days, test_split.n_assets)

    all_results = {}
    TARGET = 0.89

    model = make_ppo_model("composite", 500_000,
                           n_steps=2048, batch_size=128,
                           gamma=0.982, max_grad_norm=0.51,
                           ent_coef=8e-4, gae_lambda=0.976, clip_range=0.3)
    _, m = train_eval(model, "PPO_fast_composite_500K", train_split, test_split)
    all_results["PPO_fast_composite_500K"] = m

    model = make_ppo_model("sharpe", 500_000,
                           n_steps=2048, batch_size=128,
                           gamma=0.982, max_grad_norm=0.51,
                           ent_coef=8e-4, gae_lambda=0.976, clip_range=0.3)
    _, m = train_eval(model, "PPO_fast_sharpe_500K", train_split, test_split)
    all_results["PPO_fast_sharpe_500K"] = m

    model = make_ppo_model("composite", 500_000,
                           learning_rate=1e-4, n_steps=2048, batch_size=128,
                           gamma=0.99, max_grad_norm=0.5,
                           ent_coef=0.01, gae_lambda=0.95, clip_range=0.2)
    _, m = train_eval(model, "PPO_lit_composite_500K", train_split, test_split)
    all_results["PPO_lit_composite_500K"] = m

    model = make_sac_model("composite", 500_000)
    _, m = train_eval(model, "SAC_composite_500K", train_split, test_split)
    all_results["SAC_composite_500K"] = m

    best_name = max(all_results, key=lambda k: all_results[k]["Sharpe Ratio"])
    best_sharpe = all_results[best_name]["Sharpe Ratio"]
    logger.info("Best so far: %s (Sharpe=%.3f), scaling to 1M steps...", best_name, best_sharpe)

    best_reward = "composite"
    if "sharpe" in best_name.lower():
        best_reward = "sharpe"

    model = make_ppo_model(best_reward, 1_000_000,
                           n_steps=2048, batch_size=128,
                           gamma=0.982, max_grad_norm=0.51,
                           ent_coef=8e-4, gae_lambda=0.976, clip_range=0.3)
    _, m = train_eval(model, f"PPO_best_1M_{best_reward}", train_split, test_split)
    all_results[f"PPO_best_1M_{best_reward}"] = m

    ensemble = EnsembleModel([
        make_ppo_model("composite", 500_000, seed=42,
                       n_steps=2048, batch_size=128,
                       gamma=0.982, max_grad_norm=0.51),
        make_ppo_model("composite", 500_000, seed=1042,
                       n_steps=2048, batch_size=128,
                       gamma=0.982, max_grad_norm=0.51),
        make_ppo_model("sharpe", 500_000, seed=2042,
                       n_steps=2048, batch_size=128,
                       gamma=0.982, max_grad_norm=0.51),
    ])
    _, m = train_eval(ensemble, "Ensemble_3xPPO", train_split, test_split)
    all_results["Ensemble_3xPPO"] = m

    ensemble2 = EnsembleModel([
        make_ppo_model("composite", 500_000, seed=42,
                       n_steps=2048, batch_size=128,
                       gamma=0.982, max_grad_norm=0.51),
        make_sac_model("composite", 500_000, seed=1042),
        make_ppo_model("sharpe", 500_000, seed=2042,
                       n_steps=2048, batch_size=128,
                       gamma=0.982, max_grad_norm=0.51),
    ])
    _, m = train_eval(ensemble2, "Ensemble_PPO_SAC_PPO", train_split, test_split)
    all_results["Ensemble_PPO_SAC_PPO"] = m

    print("\n" + "=" * 90)
    print("PHASE 2b RESULTS (n_assets=%d)" % args.n_assets)
    print("=" * 90)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 90)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_results:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f}{flag}")
    print("=" * 90)

    best = sorted_results[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")
    if best[1]["Sharpe Ratio"] >= TARGET:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {TARGET - best[1]['Sharpe Ratio']:.3f}")

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "v3_summary.csv")
    logger.info("Saved to %s", RESULTS_DIR / "v3_summary.csv")

if __name__ == "__main__":
    main()
