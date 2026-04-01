from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import (
    _BaseRLModel,
    _build_price_features,
    _build_graph_tda_features,
    SAPPOModel,
)
from src.rl.environment import PortfolioEnv, SAPPOEnv
from src.rl.train_fast import filter_splits_to_common_universe, train_eval

DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "techniques"

class RewardClipPPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            max_weight=self.max_weight,
            lambda_dd=self.lambda_dd,
            lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold,
            dsr_eta=self.dsr_eta,
            reward_clip=(-0.4, 0.4),
        )
        env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class TurbulencePPO(_BaseRLModel):

    def __init__(self, turbulence_pct: float = 95.0, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)
        self.turbulence_pct = turbulence_pct

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            max_weight=self.max_weight,
            lambda_dd=self.lambda_dd,
            lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold,
            dsr_eta=self.dsr_eta,
            turbulence_threshold_pct=self.turbulence_pct,
        )
        env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class RiskSensitivePPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "risk_sensitive")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class EigCentralityPPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class RecurrentPPOModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "RecurrentPPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("use_recurrent", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
        )
        return RecurrentPPO(
            "MlpLstmPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class CombinedTechniquesPPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _make_env(self, returns, features_pa, features_global):
        env_kwargs = dict(
            returns=returns,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            max_weight=self.max_weight,
            lambda_dd=self.lambda_dd,
            lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold,
            dsr_eta=self.dsr_eta,
            reward_clip=(-0.4, 0.4),
            turbulence_threshold_pct=95.0,
        )
        env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

class BaselinePPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("total_timesteps", 500_000)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0, seed=seed, device=DEVICE,
        )

def main():
    parser = argparse.ArgumentParser(description="Train all new techniques")
    parser.add_argument("--n_assets", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--skip_graph", action="store_true",
                        help="Skip graph-based models (slow)")
    parser.add_argument("--skip_sappo", action="store_true",
                        help="Skip SAPPO (requires sentiment data)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=%d)...", args.n_assets)
    pipeline = DataPipeline(n_assets=args.n_assets)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits_to_common_universe(train_split, test_split)

    logger.info("Train: %d days, %d assets | Test: %d days, %d assets",
                train_split.n_days, train_split.n_assets,
                test_split.n_days, test_split.n_assets)

    all_results = {}
    ts = args.timesteps

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE: Standard PPO (no new techniques)")
    logger.info("=" * 70)
    model = BaselinePPO(total_timesteps=ts)
    _, metrics = train_eval(model, "Baseline_PPO", train_split, test_split)
    all_results["Baseline_PPO"] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("TECHNIQUE 1: Reward Clipping [-0.4, 0.4]")
    logger.info("=" * 70)
    model = RewardClipPPO(total_timesteps=ts)
    _, metrics = train_eval(model, "RewardClip_PPO", train_split, test_split)
    all_results["RewardClip_PPO"] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("TECHNIQUE 2: Turbulence Cash Switching (95th pct)")
    logger.info("=" * 70)
    model = TurbulencePPO(total_timesteps=ts, turbulence_pct=95.0)
    _, metrics = train_eval(model, "Turbulence_PPO", train_split, test_split)
    all_results["Turbulence_PPO"] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("TECHNIQUE 3: Risk-Sensitive Reward")
    logger.info("=" * 70)
    model = RiskSensitivePPO(total_timesteps=ts)
    _, metrics = train_eval(model, "RiskSensitive_PPO", train_split, test_split)
    all_results["RiskSensitive_PPO"] = metrics

    if not args.skip_graph:
        logger.info("\n" + "=" * 70)
        logger.info("TECHNIQUE 4: Eigenvector Centrality (full graph features)")
        logger.info("=" * 70)
        model = EigCentralityPPO(total_timesteps=ts)
        _, metrics = train_eval(model, "EigCentrality_PPO", train_split, test_split)
        all_results["EigCentrality_PPO"] = metrics
    else:
        logger.info("SKIPPING Technique 4 (graph features — use --skip_graph to skip)")

    logger.info("\n" + "=" * 70)
    logger.info("TECHNIQUE 5: RecurrentPPO (LSTM policy)")
    logger.info("=" * 70)
    model = RecurrentPPOModel(total_timesteps=ts)
    _, metrics = train_eval(model, "RecurrentPPO", train_split, test_split)
    all_results["RecurrentPPO"] = metrics

    if not args.skip_sappo:
        logger.info("\n" + "=" * 70)
        logger.info("TECHNIQUE 6: SAPPO (Sentiment-Aware PPO)")
        logger.info("=" * 70)
        model = SAPPOModel(sappo_alpha=0.1, total_timesteps=ts)
        _, metrics = train_eval(model, "SAPPO", train_split, test_split)
        all_results["SAPPO"] = metrics
    else:
        logger.info("SKIPPING Technique 6 (SAPPO — use --skip_sappo to skip)")

    logger.info("\n" + "=" * 70)
    logger.info("COMBINED: Reward Clip + Turbulence Switching")
    logger.info("=" * 70)
    model = CombinedTechniquesPPO(total_timesteps=ts)
    _, metrics = train_eval(model, "Combined_ClipTurb", train_split, test_split)
    all_results["Combined_ClipTurb"] = metrics

    print("\n" + "=" * 90)
    print("ALL TECHNIQUES — RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<25s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | "
          f"{'Calmar':>7s} | {'Sortino':>7s} | {'vs Base':>8s}")
    print("-" * 90)

    baseline_sharpe = all_results.get("Baseline_PPO", {}).get("Sharpe Ratio", 0.0)
    sorted_final = sorted(all_results.items(),
                          key=lambda x: x[1]["Sharpe Ratio"], reverse=True)

    for name, m in sorted_final:
        delta = m["Sharpe Ratio"] - baseline_sharpe
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<23s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f} | "
              f"{sign}{delta:7.3f}")
    print("=" * 90)

    summary_df = pd.DataFrame(all_results).T
    summary_df["delta_sharpe_vs_baseline"] = summary_df["Sharpe Ratio"] - baseline_sharpe
    summary_df.to_csv(RESULTS_DIR / "techniques_summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "techniques_summary.csv")

    best = sorted_final[0]
    print(f"\nBest technique: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")

if __name__ == "__main__":
    main()
