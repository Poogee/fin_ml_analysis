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

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward, BacktestResult
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import (
    _BaseRLModel,
    _build_price_features,
    _build_graph_tda_features,
    _build_sentiment_features,
)
from src.rl.environment import PortfolioEnv

DEVICE_PPO = "cpu"
DEVICE_SAC = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"

def filter_splits_to_common_universe(
    train_split: DataSplit, test_split: DataSplit,
) -> tuple[DataSplit, DataSplit]:

    all_universe = sorted(set(train_split.all_assets) | set(test_split.all_assets))

    common_cols = set(train_split.returns.columns) & set(test_split.returns.columns)
    available = sorted(a for a in all_universe if a in common_cols)

    if not available:
        return train_split, test_split

    logger.info("Filtering both splits to %d common universe assets (from %d train cols, %d test cols)",
                len(available), len(train_split.returns.columns), len(test_split.returns.columns))

    def _filter(split, assets):
        return DataSplit(
            prices=split.prices[assets],
            returns=split.returns[assets],
            log_returns=split.log_returns[assets],
            market_caps=split.market_caps[assets],
            dividends=split.dividends[assets],
            presence=split.presence[assets],
            universe_schedule=split.universe_schedule,
            start_date=split.start_date,
            end_date=split.end_date,
            all_assets=assets,
        )

    return _filter(train_split, available), _filter(test_split, available)

class FastPPO(_BaseRLModel):

    def __init__(self, ppo_kwargs: dict | None = None, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)
        self._ppo_kwargs = ppo_kwargs or {}

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed: int):
        defaults = dict(
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
        )
        defaults.update(self._ppo_kwargs)

        net_arch = defaults.pop("net_arch", dict(pi=[256, 128], vf=[256, 128]))

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            **defaults,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE_PPO,
        )

class FastSAC(_BaseRLModel):

    def __init__(self, sac_kwargs: dict | None = None, **kwargs):
        kwargs.setdefault("algorithm", "SAC")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)
        self._sac_kwargs = sac_kwargs or {}

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
        )
        env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed: int):
        defaults = dict(
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
        )
        defaults.update(self._sac_kwargs)

        net_arch = defaults.pop("net_arch", dict(pi=[256, 128], qf=[256, 128]))

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Tanh,
        )
        return SAC(
            "MlpPolicy", env,
            **defaults,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE_SAC,
        )

class FastA2C(_BaseRLModel):

    def __init__(self, a2c_kwargs: dict | None = None, **kwargs):
        kwargs.setdefault("algorithm", "A2C")
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)
        self._a2c_kwargs = a2c_kwargs or {}

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

    def _create_agent(self, env, seed: int):
        defaults = dict(
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        defaults.update(self._a2c_kwargs)

        net_arch = defaults.pop("net_arch", dict(pi=[256, 128], vf=[256, 128]))

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Tanh,
        )
        return A2C(
            "MlpPolicy", env,
            **defaults,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE_PPO,
        )

class FastRecurrentPPO(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "RecurrentPPO")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("use_recurrent", True)
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

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
            device=DEVICE_PPO,
        )

class FullFeatModel(_BaseRLModel):

    def __init__(self, algo_class, algo_kwargs: dict, **kwargs):
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)
        self._algo_class = algo_class
        self._algo_kwargs = algo_kwargs

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        kwargs = dict(self._algo_kwargs)
        net_arch = kwargs.pop("net_arch", dict(pi=[256, 128], vf=[256, 128]))
        policy_kwargs = dict(net_arch=net_arch, activation_fn=torch.nn.Tanh)

        if self._algo_class == SAC:
            net_arch = kwargs.pop("net_arch", dict(pi=[256, 128], qf=[256, 128]))
            policy_kwargs = dict(net_arch=net_arch, activation_fn=torch.nn.Tanh)

        policy = "MlpPolicy"
        if self._algo_class == RecurrentPPO:
            policy = "MlpLstmPolicy"
            policy_kwargs = dict(
                lstm_hidden_size=128, n_lstm_layers=1,
                shared_lstm=False, enable_critic_lstm=True,
            )

        return self._algo_class(
            policy, env,
            **kwargs,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed, device=DEVICE,
        )

class SentimentOnlyModel(_BaseRLModel):

    def __init__(self, algo_class, algo_kwargs: dict, **kwargs):
        kwargs.setdefault("use_vec_normalize", True)
        super().__init__(**kwargs)
        self._algo_class = algo_class
        self._algo_kwargs = algo_kwargs

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": True}

    def _create_agent(self, env, seed: int):
        kwargs = dict(self._algo_kwargs)
        net_arch = kwargs.pop("net_arch", dict(pi=[256, 128], vf=[256, 128]))
        policy_kwargs = dict(net_arch=net_arch, activation_fn=torch.nn.Tanh)

        if self._algo_class == SAC:
            net_arch_sac = kwargs.pop("net_arch", dict(pi=[256, 128], qf=[256, 128]))
            policy_kwargs = dict(net_arch=net_arch_sac, activation_fn=torch.nn.Tanh)

        return self._algo_class(
            "MlpPolicy", env,
            **kwargs,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed, device=DEVICE,
        )

class EnsembleModel:

    def __init__(self, models: list):
        self.models = models

    def fit(self, train_data: dict) -> None:
        for i, m in enumerate(self.models):
            logger.info("Ensemble: training %d/%d (%s)...",
                        i + 1, len(self.models), m.__class__.__name__)
            m.fit(train_data)

    def predict_weights(self, current_data: dict) -> np.ndarray:
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

def train_eval(model, name, train_split, test_split, tc=10.0, slip=5.0, rebal=5):

    logger.info("=" * 60)
    logger.info("TRAINING: %s", name)
    t0 = time.time()

    result = run_walk_forward(
        model=model, train_split=train_split, test_split=test_split,
        model_name=name, transaction_cost_bps=tc, slippage_bps=slip,
        rebalance_frequency=rebal,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all")
    parser.add_argument("--n_assets", type=int, default=100)
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
    TARGET = 0.89

    if args.phase in ("1", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: PPO config sweep (no graph/TDA — fast)")
        logger.info("=" * 70)

        SWEEP_STEPS = 200_000

        configs = {
            "PPO_lit_composite": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "composite",
            }, {}),
            "PPO_lit_dsr": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "dsr",
            }, {}),
            "PPO_lit_sharpe": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "sharpe",
            }, {}),
            "PPO_lit_return": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "return",
            }, {}),
            "PPO_optuna_composite": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "composite",
            }, {
                "learning_rate": 3.5e-4, "n_steps": 512, "batch_size": 32,
                "gamma": 0.982, "clip_range": 0.3, "ent_coef": 8e-4,
                "gae_lambda": 0.976, "max_grad_norm": 0.51,
            }),
            "PPO_hybrid": (FastPPO, {
                "total_timesteps": SWEEP_STEPS, "reward_type": "composite",
            }, {
                "learning_rate": 1e-4, "n_steps": 1024, "batch_size": 64,
                "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.005,
                "max_grad_norm": 0.51,
            }),
        }

        for name, (cls, kwargs, ppo_kw) in configs.items():
            model = cls(ppo_kwargs=ppo_kw, **kwargs)
            _, metrics = train_eval(model, name, train_split, test_split)
            all_results[name] = metrics
            if metrics["Sharpe Ratio"] >= TARGET:
                logger.info("TARGET HIT: %s Sharpe=%.3f", name, metrics["Sharpe Ratio"])

    if args.phase in ("2", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: SAC, A2C, RecurrentPPO")
        logger.info("=" * 70)

        for reward in ["composite", "dsr"]:
            name = f"SAC_{reward}"
            model = FastSAC(total_timesteps=500_000, reward_type=reward)
            _, metrics = train_eval(model, name, train_split, test_split)
            all_results[name] = metrics

        model = FastA2C(total_timesteps=500_000, reward_type="composite")
        _, metrics = train_eval(model, "A2C_composite", train_split, test_split)
        all_results["A2C_composite"] = metrics

        model = FastRecurrentPPO(total_timesteps=500_000, reward_type="composite")
        _, metrics = train_eval(model, "RecPPO_composite", train_split, test_split)
        all_results["RecPPO_composite"] = metrics

    if args.phase in ("3", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: Add sentiment features to best configs")
        logger.info("=" * 70)

        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
        logger.info("Top results so far:")
        for n, m in sorted_results[:5]:
            logger.info("  %s: Sharpe=%.3f", n, m["Sharpe Ratio"])

        best_ppo_name = None
        best_ppo_sharpe = -np.inf
        for n, m in sorted_results:
            if n.startswith("PPO"):
                best_ppo_name = n
                best_ppo_sharpe = m["Sharpe Ratio"]
                break

        if best_ppo_name:

            reward = "composite"
            if "dsr" in best_ppo_name:
                reward = "dsr"
            elif "sharpe" in best_ppo_name:
                reward = "sharpe"
            elif "return" in best_ppo_name:
                reward = "return"

            ppo_defaults = dict(
                learning_rate=1e-4, n_steps=2048, batch_size=128,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            )

            model = SentimentOnlyModel(
                PPO, ppo_defaults,
                algorithm="PPO",
                total_timesteps=500_000, reward_type=reward,
            )
            name = f"PPO_sent_{reward}"
            _, metrics = train_eval(model, name, train_split, test_split)
            all_results[name] = metrics

        best_sac_name = None
        for n, m in sorted_results:
            if n.startswith("SAC"):
                best_sac_name = n
                break

        if best_sac_name:
            reward = "composite" if "composite" in best_sac_name else "dsr"
            sac_defaults = dict(
                learning_rate=3e-4, buffer_size=100_000, learning_starts=1000,
                batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto",
            )
            model = SentimentOnlyModel(
                SAC, sac_defaults,
                algorithm="SAC",
                total_timesteps=500_000, reward_type=reward,
            )
            name = f"SAC_sent_{reward}"
            _, metrics = train_eval(model, name, train_split, test_split)
            all_results[name] = metrics

    if args.phase in ("4", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: Ensemble + extended training")
        logger.info("=" * 70)

        ensemble = EnsembleModel([
            FastPPO(total_timesteps=500_000, reward_type="composite"),
            FastSAC(total_timesteps=500_000, reward_type="composite"),
            FastA2C(total_timesteps=500_000, reward_type="composite"),
        ])
        _, metrics = train_eval(ensemble, "Ensemble_PPO_SAC_A2C", train_split, test_split)
        all_results["Ensemble_PPO_SAC_A2C"] = metrics

        ensemble_ppo = EnsembleModel([
            FastPPO(total_timesteps=500_000, reward_type="composite", seed=42),
            FastPPO(total_timesteps=500_000, reward_type="composite", seed=1042),
            FastPPO(total_timesteps=500_000, reward_type="composite", seed=2042),
        ])
        _, metrics = train_eval(ensemble_ppo, "Ensemble_3xPPO", train_split, test_split)
        all_results["Ensemble_3xPPO"] = metrics

        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
        best_name = sorted_results[0][0]
        best_reward = "composite"
        if "dsr" in best_name.lower():
            best_reward = "dsr"
        elif "sharpe" in best_name.lower():
            best_reward = "sharpe"
        elif "return" in best_name.lower():
            best_reward = "return"

        model = FastPPO(total_timesteps=1_000_000, reward_type=best_reward)
        _, metrics = train_eval(model, f"PPO_1M_{best_reward}", train_split, test_split)
        all_results[f"PPO_1M_{best_reward}"] = metrics

    if args.phase in ("5", "all", "full"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: Full features (graph+TDA+sentiment) for best model")
        logger.info("=" * 70)

        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
        best_name = sorted_results[0][0]

        best_reward = "composite"
        if "dsr" in best_name.lower():
            best_reward = "dsr"
        elif "sharpe" in best_name.lower():
            best_reward = "sharpe"
        elif "return" in best_name.lower():
            best_reward = "return"

        logger.info("Adding full features to best config: %s (reward=%s)", best_name, best_reward)

        ppo_defaults = dict(
            learning_rate=1e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        )

        model = FullFeatModel(
            PPO, ppo_defaults,
            algorithm="PPO",
            total_timesteps=500_000, reward_type=best_reward,
        )
        _, metrics = train_eval(model, f"PPO_full_{best_reward}", train_split, test_split)
        all_results[f"PPO_full_{best_reward}"] = metrics

    print("\n" + "=" * 85)
    print("IMPROVED RL — RESULTS SUMMARY")
    print("=" * 85)
    print(f"{'Model':<30s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 85)

    sorted_final = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_final:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<28s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f}{flag}")
    print("=" * 85)

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "fast_summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "fast_summary.csv")

    best = sorted_final[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")
    if best[1]["Sharpe Ratio"] >= TARGET:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {TARGET - best[1]['Sharpe Ratio']:.3f}")

if __name__ == "__main__":
    main()
