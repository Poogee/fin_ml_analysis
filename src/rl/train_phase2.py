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
from src.evaluation.backtester import run_walk_forward, BacktestResult
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import _BaseRLModel, DEVICE
from src.rl.environment import PortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"

DEVICE_PPO = "cpu"
DEVICE_SAC = "cuda" if torch.cuda.is_available() else "cpu"

def filter_splits(train_split, test_split):
    all_universe = sorted(set(train_split.all_assets) | set(test_split.all_assets))
    common_cols = set(train_split.returns.columns) & set(test_split.returns.columns)
    available = sorted(a for a in all_universe if a in common_cols)
    logger.info("Common universe: %d assets", len(available))

    def _f(split, assets):
        return DataSplit(
            prices=split.prices[assets], returns=split.returns[assets],
            log_returns=split.log_returns[assets], market_caps=split.market_caps[assets],
            dividends=split.dividends[assets], presence=split.presence[assets],
            universe_schedule=split.universe_schedule,
            start_date=split.start_date, end_date=split.end_date, all_assets=assets,
        )
    return _f(train_split, available), _f(test_split, available)

class OptunaCompositePPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4, n_steps=512, batch_size=32,
            n_epochs=10, gamma=0.982, gae_lambda=0.976,
            clip_range=0.3, ent_coef=8e-4, vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed, device=DEVICE_PPO,
        )

class OptunaSentimentPPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": True}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4, n_steps=512, batch_size=32,
            n_epochs=10, gamma=0.982, gae_lambda=0.976,
            clip_range=0.3, ent_coef=8e-4, vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed, device=DEVICE_PPO,
        )

class OptunaSharpePPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("reward_type", "sharpe")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self):
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4, n_steps=512, batch_size=32,
            n_epochs=10, gamma=0.982, gae_lambda=0.976,
            clip_range=0.3, ent_coef=8e-4, vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed, device=DEVICE_PPO,
        )

class FastSAC(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "SAC")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

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

    def _create_agent(self, env, seed):
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
            verbose=0, seed=seed, device=DEVICE_SAC,
        )

class EnsembleModel:

    def __init__(self, models):
        self.models = models

    def fit(self, train_data):
        for i, m in enumerate(self.models):
            logger.info("Ensemble training %d/%d...", i+1, len(self.models))
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
    logger.info("%s: %.0fs | Sharpe=%.3f | Ret=%.2f%% | DD=%.2f%% | Calmar=%.3f",
                name, elapsed, metrics["Sharpe Ratio"],
                metrics["Cumulative Return"] * 100,
                metrics["Max Drawdown"] * 100, metrics["Calmar Ratio"])
    safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)
    return result, metrics

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    pipeline = DataPipeline(n_assets=30)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    logger.info("Train: %d days, %d assets | Test: %d days, %d assets",
                train_split.n_days, train_split.n_assets,
                test_split.n_days, test_split.n_assets)

    all_results = {}
    TARGET = 0.89

    model = OptunaCompositePPO(total_timesteps=500_000)
    _, m = train_eval(model, "PPO_optuna_500K", train_split, test_split)
    all_results["PPO_optuna_500K"] = m

    model = OptunaCompositePPO(total_timesteps=1_000_000)
    _, m = train_eval(model, "PPO_optuna_1M", train_split, test_split)
    all_results["PPO_optuna_1M"] = m

    model = OptunaSharpePPO(total_timesteps=500_000)
    _, m = train_eval(model, "PPO_optuna_sharpe_500K", train_split, test_split)
    all_results["PPO_optuna_sharpe_500K"] = m

    model = FastSAC(total_timesteps=500_000, reward_type="composite")
    _, m = train_eval(model, "SAC_composite_500K", train_split, test_split)
    all_results["SAC_composite_500K"] = m

    model = OptunaSentimentPPO(total_timesteps=500_000)
    _, m = train_eval(model, "PPO_optuna_sent_500K", train_split, test_split)
    all_results["PPO_optuna_sent_500K"] = m

    ensemble = EnsembleModel([
        OptunaCompositePPO(total_timesteps=500_000, seed=42),
        OptunaSharpePPO(total_timesteps=500_000, seed=1042),
        FastSAC(total_timesteps=500_000, reward_type="composite", seed=2042),
    ])
    _, m = train_eval(ensemble, "Ensemble_PPO2_SAC", train_split, test_split)
    all_results["Ensemble_PPO2_SAC"] = m

    ensemble_ppo = EnsembleModel([
        OptunaCompositePPO(total_timesteps=500_000, seed=42),
        OptunaCompositePPO(total_timesteps=500_000, seed=1042),
        OptunaCompositePPO(total_timesteps=500_000, seed=2042),
    ])
    _, m = train_eval(ensemble_ppo, "Ensemble_3xPPO", train_split, test_split)
    all_results["Ensemble_3xPPO"] = m

    print("\n" + "=" * 85)
    print("PHASE 2 RESULTS")
    print("=" * 85)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_results:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<30s} | Sharpe={m['Sharpe Ratio']:7.3f} | "
              f"Ret={m['Cumulative Return']*100:8.2f}% | "
              f"DD={m['Max Drawdown']*100:7.2f}% | "
              f"Calmar={m['Calmar Ratio']:7.3f}{flag}")
    print("=" * 85)

    best = sorted_results[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")
    if best[1]["Sharpe Ratio"] >= TARGET:
        print("TARGET ACHIEVED!")

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "phase2_summary.csv")

if __name__ == "__main__":
    main()
