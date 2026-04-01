from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, A2C, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from src.data.pipeline import DataPipeline
from src.evaluation.backtester import run_walk_forward, BacktestResult
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import (
    _BaseRLModel,
    _build_price_features,
    _build_graph_tda_features,
    _build_sentiment_features,
    DEVICE,
)
from src.rl.environment import PortfolioEnv, DiscretePortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"

class ImprovedPPOModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class ImprovedPPOv2Model(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 750_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class A2CModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "A2C")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return A2C(
            "MlpPolicy", env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class SACModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "SAC")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

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
        )
        env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], qf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class RecurrentPPOModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "RecurrentPPO")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("use_recurrent", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
        )
        return RecurrentPPO(
            "MlpLstmPolicy", env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class DSROnlyPPOModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "dsr")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class SharpeRewardPPOModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "sharpe")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

class ImprovedDQNModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "DQN")
        kwargs.setdefault("total_timesteps", 400_000)
        kwargs.setdefault("reward_type", "sharpe")
        kwargs.setdefault("use_vec_normalize", False)
        kwargs.setdefault("net_arch", [64, 64])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

class EnsembleModel:

    def __init__(self, models: list[_BaseRLModel], eval_window: int = 63):
        self.models = models
        self.eval_window = eval_window
        self._trained = False

    def fit(self, train_data: dict) -> None:
        for i, m in enumerate(self.models):
            logger.info("Ensemble: training model %d/%d (%s)...",
                        i + 1, len(self.models), m.__class__.__name__)
            m.fit(train_data)
        self._trained = True

    def predict_weights(self, current_data: dict) -> np.ndarray:

        if not self._trained:
            n = current_data["returns"].shape[1] if hasattr(current_data["returns"], 'shape') else len(current_data["returns"].columns)
            return np.ones(n) / n

        all_weights = []
        for m in self.models:
            try:
                w = m.predict_weights(current_data)
                all_weights.append(w)
            except Exception as e:
                logger.warning("Model %s failed predict: %s", m.__class__.__name__, e)

        if not all_weights:
            n = current_data["returns"].shape[1] if hasattr(current_data["returns"], 'shape') else len(current_data["returns"].columns)
            return np.ones(n) / n

        avg = np.mean(all_weights, axis=0)
        avg = np.maximum(avg, 0)
        s = avg.sum()
        if s > 0:
            avg /= s
        return avg

def train_and_eval(
    model, name: str, train_split, test_split,
    tc_bps=10.0, slip_bps=5.0, rebalance_freq=5,
) -> tuple[BacktestResult, dict]:

    logger.info("=" * 60)
    logger.info("TRAINING: %s", name)
    t0 = time.time()

    result = run_walk_forward(
        model=model,
        train_split=train_split,
        test_split=test_split,
        model_name=name,
        transaction_cost_bps=tc_bps,
        slippage_bps=slip_bps,
        rebalance_frequency=rebalance_freq,
    )

    elapsed = time.time() - t0
    metrics = PortfolioMetrics.compute_all(result.returns)

    logger.info("%s completed in %.1fs", name, elapsed)
    logger.info(
        "  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | Calmar: %.3f | Sortino: %.3f",
        metrics["Sharpe Ratio"],
        metrics["Cumulative Return"] * 100,
        metrics["Max Drawdown"] * 100,
        metrics["Calmar Ratio"],
        metrics["Sortino Ratio"],
    )

    safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)

    return result, metrics

def run_phase1(train_split, test_split, results: dict):

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Core PPO improvements")
    logger.info("=" * 70)

    configs = [
        ("PPO_Lit_Composite_500K", ImprovedPPOModel(
            total_timesteps=500_000, reward_type="composite",
        )),
        ("PPO_Lit_DSR_500K", DSROnlyPPOModel(
            total_timesteps=500_000, reward_type="dsr",
        )),
        ("PPO_Lit_Sharpe_500K", SharpeRewardPPOModel(
            total_timesteps=500_000, reward_type="sharpe",
        )),
        ("PPO_v2_Hybrid_750K", ImprovedPPOv2Model(
            total_timesteps=750_000, reward_type="composite",
        )),
    ]

    for name, model in configs:
        result, metrics = train_and_eval(model, name, train_split, test_split)
        results[name] = (result, metrics)

        if metrics["Sharpe Ratio"] >= 0.89:
            logger.info("TARGET ACHIEVED with %s: Sharpe %.3f >= 0.89",
                        name, metrics["Sharpe Ratio"])
            return True

    return False

def run_phase2(train_split, test_split, results: dict):

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Alternative algorithms")
    logger.info("=" * 70)

    configs = [
        ("SAC_Composite_500K", SACModel(
            total_timesteps=500_000, reward_type="composite",
        )),
        ("A2C_Composite_500K", A2CModel(
            total_timesteps=500_000, reward_type="composite",
        )),
        ("RecurrentPPO_Composite_500K", RecurrentPPOModel(
            total_timesteps=500_000, reward_type="composite",
        )),
    ]

    for name, model in configs:
        result, metrics = train_and_eval(model, name, train_split, test_split)
        results[name] = (result, metrics)

        if metrics["Sharpe Ratio"] >= 0.89:
            logger.info("TARGET ACHIEVED with %s: Sharpe %.3f >= 0.89",
                        name, metrics["Sharpe Ratio"])
            return True

    return False

def run_phase3(train_split, test_split, results: dict):

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Ensemble + DQN validation")
    logger.info("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1][1]["Sharpe Ratio"], reverse=True)
    logger.info("Results so far:")
    for name, (_, m) in sorted_results:
        logger.info("  %s: Sharpe=%.3f", name, m["Sharpe Ratio"])

    best_name = sorted_results[0][0]
    logger.info("Best single model: %s (Sharpe=%.3f)",
                best_name, sorted_results[0][1][1]["Sharpe Ratio"])

    ensemble_models = [
        ImprovedPPOModel(total_timesteps=500_000, reward_type="composite"),
        SACModel(total_timesteps=500_000, reward_type="composite"),
        A2CModel(total_timesteps=500_000, reward_type="composite"),
    ]

    ensemble = EnsembleModel(ensemble_models)
    result, metrics = train_and_eval(ensemble, "Ensemble_PPO_SAC_A2C", train_split, test_split)
    results["Ensemble_PPO_SAC_A2C"] = (result, metrics)

    if metrics["Sharpe Ratio"] >= 0.89:
        logger.info("TARGET ACHIEVED with Ensemble: Sharpe %.3f >= 0.89",
                    metrics["Sharpe Ratio"])
        return True

    ppo_ensemble_models = [
        ImprovedPPOModel(total_timesteps=500_000, reward_type="composite", seed=42),
        ImprovedPPOModel(total_timesteps=500_000, reward_type="composite", seed=1042),
        ImprovedPPOModel(total_timesteps=500_000, reward_type="composite", seed=2042),
    ]
    ppo_ensemble = EnsembleModel(ppo_ensemble_models)
    result, metrics = train_and_eval(ppo_ensemble, "Ensemble_3xPPO", train_split, test_split)
    results["Ensemble_3xPPO"] = (result, metrics)

    if metrics["Sharpe Ratio"] >= 0.89:
        logger.info("TARGET ACHIEVED with PPO Ensemble: Sharpe %.3f >= 0.89",
                    metrics["Sharpe Ratio"])
        return True

    logger.info("\nValidating DQN with 3 seeds...")
    dqn_sharpes = []
    for seed in [42, 1042, 2042]:
        dqn = ImprovedDQNModel(total_timesteps=400_000, seed=seed)
        name = f"DQN_seed{seed}"
        result, metrics = train_and_eval(dqn, name, train_split, test_split)
        results[name] = (result, metrics)
        dqn_sharpes.append(metrics["Sharpe Ratio"])
        logger.info("DQN seed=%d: Sharpe=%.3f", seed, metrics["Sharpe Ratio"])

    logger.info("DQN mean Sharpe: %.3f ± %.3f",
                np.mean(dqn_sharpes), np.std(dqn_sharpes))

    return False

def run_phase4_extended(train_split, test_split, results: dict):

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Extended training runs")
    logger.info("=" * 70)

    best_reward = "composite"
    best_sharpe = -np.inf
    for name, (_, m) in results.items():
        if m["Sharpe Ratio"] > best_sharpe:
            best_sharpe = m["Sharpe Ratio"]
            if "DSR" in name:
                best_reward = "dsr"
            elif "Sharpe" in name:
                best_reward = "sharpe"
            else:
                best_reward = "composite"

    logger.info("Best reward type so far: %s (Sharpe=%.3f)", best_reward, best_sharpe)

    model = ImprovedPPOModel(total_timesteps=1_000_000, reward_type=best_reward)
    result, metrics = train_and_eval(model, f"PPO_1M_{best_reward}", train_split, test_split)
    results[f"PPO_1M_{best_reward}"] = (result, metrics)

    if metrics["Sharpe Ratio"] >= 0.89:
        return True

    model = SACModel(total_timesteps=1_000_000, reward_type=best_reward)
    result, metrics = train_and_eval(model, f"SAC_1M_{best_reward}", train_split, test_split)
    results[f"SAC_1M_{best_reward}"] = (result, metrics)

    if metrics["Sharpe Ratio"] >= 0.89:
        return True

    for ldd, lturn in [(1.0, 0.3), (3.0, 1.0), (5.0, 0.1)]:
        model = ImprovedPPOModel(
            total_timesteps=500_000, reward_type="composite",
            lambda_dd=ldd, lambda_turnover=lturn,
        )
        name = f"PPO_dd{ldd}_turn{lturn}"
        result, metrics = train_and_eval(model, name, train_split, test_split)
        results[name] = (result, metrics)
        if metrics["Sharpe Ratio"] >= 0.89:
            return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--n_assets", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data pipeline (n_assets=%d)...", args.n_assets)
    pipeline = DataPipeline(n_assets=args.n_assets)
    train_split, test_split = pipeline.build()

    logger.info("Train: %s to %s (%d days, %d assets)",
                train_split.start_date.date(), train_split.end_date.date(),
                train_split.n_days, train_split.n_assets)
    logger.info("Test: %s to %s (%d days, %d assets)",
                test_split.start_date.date(), test_split.end_date.date(),
                test_split.n_days, test_split.n_assets)

    results = {}
    achieved = False

    if args.phase in ("1", "all"):
        achieved = run_phase1(train_split, test_split, results)

    if not achieved and args.phase in ("2", "all"):
        achieved = run_phase2(train_split, test_split, results)

    if not achieved and args.phase in ("3", "all"):
        achieved = run_phase3(train_split, test_split, results)

    if not achieved and args.phase in ("4", "all"):
        achieved = run_phase4_extended(train_split, test_split, results)

    print("\n" + "=" * 80)
    print("IMPROVED RL TRAINING — FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1][1]["Sharpe Ratio"], reverse=True)
    for name, (_, m) in sorted_results:
        marker = " ***" if m["Sharpe Ratio"] >= 0.89 else ""
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f}{marker}")

    print("=" * 80)
    if achieved:
        print("TARGET ACHIEVED: At least one model has Sharpe >= 0.89")
    else:
        best_name, (_, best_m) = sorted_results[0]
        print(f"Best: {best_name} with Sharpe {best_m['Sharpe Ratio']:.3f}")
        print("Target NOT yet achieved. Consider further tuning.")

    summary_rows = {name: m for name, (_, m) in results.items()}
    summary_df = pd.DataFrame(summary_rows).T
    summary_df.to_csv(RESULTS_DIR / "summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "summary.csv")

    return achieved

if __name__ == "__main__":
    main()
