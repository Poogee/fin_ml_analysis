from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline
from src.rl.environment import PortfolioEnv, DiscretePortfolioEnv
from src.rl.agent import _build_price_features, _build_graph_tda_features
from src.evaluation.metrics import PortfolioMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DEVICE_PPO = "cpu"
DEVICE_DQN = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent.parent
TUNING_DIR = PROJECT_ROOT / "configs" / "tuning_results"

def _build_env_data(returns: np.ndarray, use_graph: bool = False,
                    reward_type: str = "return"):

    price_feats = _build_price_features(returns, lookback=20)
    features_pa = price_feats
    features_global = None

    if use_graph:
        graph_pa, graph_global = _build_graph_tda_features(
            returns, corr_window=60, graph_k=10,
            diffusion_times=[0.5, 1.0, 5.0], n_clusters=4,
        )
        features_pa = np.concatenate([features_pa, graph_pa], axis=2)
        features_global = graph_global

    return dict(
        returns=returns,
        features_per_asset=features_pa,
        features_global=features_global,
        lookback=20,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        max_weight=0.05,
        reward_type=reward_type,
    )

def _evaluate_agent(agent, val_returns: np.ndarray, env_class,
                    use_graph: bool = False, vec_normalize=None,
                    reward_type: str = "return") -> dict:

    env_data = _build_env_data(val_returns, use_graph=use_graph,
                               reward_type=reward_type)

    if vec_normalize is not None:
        val_vec = DummyVecEnv([lambda: env_class(**env_data)])
        val_norm = VecNormalize(val_vec, norm_obs=True, norm_reward=False,
                                clip_obs=10.0)

        val_norm.obs_rms = vec_normalize.obs_rms
        val_norm.ret_rms = vec_normalize.ret_rms
        val_norm.training = False
        val_norm.norm_reward = False

        obs = val_norm.reset()
        all_rets = []
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, dones, infos = val_norm.step(action)
            all_rets.append(infos[0]["portfolio_return"])
            done = dones[0]
    else:
        env = env_class(**env_data)
        obs, _ = env.reset()
        all_rets = []
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            all_rets.append(info["portfolio_return"])
            done = terminated or truncated

    if len(all_rets) < 20:
        return {"sharpe": -10.0, "cumulative_return": -1.0, "max_drawdown": 1.0}

    rets = pd.Series(all_rets)
    return {
        "sharpe": float(PortfolioMetrics.sharpe_ratio(rets)),
        "cumulative_return": float(PortfolioMetrics.cumulative_return(rets)),
        "max_drawdown": float(PortfolioMetrics.max_drawdown(rets)),
        "n_val_days": len(all_rets),
    }

def objective_ppo(trial: optuna.Trial, train_returns: np.ndarray,
                  val_returns: np.ndarray, use_graph: bool) -> float:

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = 1.0 - trial.suggest_float("one_minus_gamma", 0.001, 0.05, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    n_steps_pow = trial.suggest_int("n_steps_pow", 8, 11)
    n_steps = 2 ** n_steps_pow
    batch_size_pow = trial.suggest_int("batch_size_pow", 5, 7)
    batch_size = min(2 ** batch_size_pow, n_steps)
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    gae_lambda = 1.0 - trial.suggest_float("one_minus_gae", 0.01, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    total_timesteps = trial.suggest_int("total_timesteps", 150_000, 400_000, step=50_000)
    reward_type = trial.suggest_categorical("reward_type", ["return", "sharpe"])

    try:
        env_data = _build_env_data(train_returns, use_graph=use_graph,
                                   reward_type=reward_type)

        def make_env():
            return PortfolioEnv(**env_data)

        env = DummyVecEnv([make_env])

        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        policy_kwargs = dict(
            net_arch=dict(
                pi=[hidden_size, hidden_size],
                vf=[hidden_size, hidden_size],
            ),
        )

        agent = PPO(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=trial.number + 42,
            device=DEVICE_PPO,
        )

        agent.learn(total_timesteps=total_timesteps)

        metrics = _evaluate_agent(
            agent, val_returns, PortfolioEnv, use_graph=use_graph,
            vec_normalize=env, reward_type=reward_type,
        )

        sharpe = metrics["sharpe"]
        trial.set_user_attr("cumulative_return", metrics["cumulative_return"])
        trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
        trial.set_user_attr("n_val_days", metrics.get("n_val_days", 0))

        logger.info(
            "Trial %d: Sharpe=%.3f, CumRet=%.2f%%, MaxDD=%.2f%% (ts=%dk, lr=%.1e, arch=%d)",
            trial.number, sharpe,
            metrics["cumulative_return"] * 100,
            metrics["max_drawdown"] * 100,
            total_timesteps // 1000, lr, hidden_size,
        )

        return sharpe

    except Exception as e:
        logger.error("Trial %d failed: %s", trial.number, e)
        return -10.0

def objective_dqn(trial: optuna.Trial, train_returns: np.ndarray,
                  val_returns: np.ndarray, use_graph: bool) -> float:

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = 1.0 - trial.suggest_float("one_minus_gamma", 0.001, 0.05, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10_000, 100_000, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size_pow", 5, 8)
    target_update = trial.suggest_int("target_update_interval", 200, 2000, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.4)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    learning_starts = trial.suggest_int("learning_starts", 500, 5000, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    total_timesteps = trial.suggest_int("total_timesteps", 100_000, 400_000, step=50_000)
    reward_type = trial.suggest_categorical("reward_type", ["return", "sharpe"])

    try:
        env_data = _build_env_data(train_returns, use_graph=use_graph,
                                   reward_type=reward_type)

        def make_env():
            return DiscretePortfolioEnv(**env_data)

        env = DummyVecEnv([make_env])

        policy_kwargs = dict(
            net_arch=[hidden_size, hidden_size],
        )

        agent = DQN(
            "MlpPolicy", env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=trial.number + 42,
            device=DEVICE_DQN,
        )

        agent.learn(total_timesteps=total_timesteps)

        metrics = _evaluate_agent(
            agent, val_returns, DiscretePortfolioEnv, use_graph=use_graph,
            reward_type=reward_type,
        )

        sharpe = metrics["sharpe"]
        trial.set_user_attr("cumulative_return", metrics["cumulative_return"])
        trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
        trial.set_user_attr("n_val_days", metrics.get("n_val_days", 0))

        logger.info(
            "Trial %d: Sharpe=%.3f, CumRet=%.2f%%, MaxDD=%.2f%% (ts=%dk, lr=%.1e, arch=%d)",
            trial.number, sharpe,
            metrics["cumulative_return"] * 100,
            metrics["max_drawdown"] * 100,
            total_timesteps // 1000, lr, hidden_size,
        )

        return sharpe

    except Exception as e:
        logger.error("Trial %d failed: %s", trial.number, e)
        return -10.0

def _param_importance(study: optuna.Study) -> dict:

    try:
        importance = optuna.importance.get_param_importances(study)
        return importance
    except Exception:
        return {}

def prepare_data(n_assets: int = 100, tune_assets: int | None = None,
                  use_graph: bool = False):

    logger.info("Loading data pipeline (n_assets=%d)...", n_assets)
    pipeline = DataPipeline(n_assets=n_assets)
    train_split, test_split = pipeline.build()

    universe_assets = train_split.all_assets
    logger.info("Universe assets for training: %d", len(universe_assets))

    if tune_assets is not None and tune_assets < len(universe_assets):
        rng = np.random.RandomState(42)
        universe_assets = sorted(rng.choice(
            universe_assets, size=tune_assets, replace=False,
        ))
        logger.info("Subsampled to %d assets for tuning", len(universe_assets))

    train_returns = train_split.returns[universe_assets].values
    train_returns = np.nan_to_num(train_returns, nan=0.0).astype(np.float32)

    split_idx = int(len(train_returns) * 0.7)
    train_sub = train_returns[:split_idx]
    val_sub = train_returns[split_idx:]

    logger.info(
        "Train subset: %d days, Val subset: %d days, Assets: %d",
        len(train_sub), len(val_sub), train_returns.shape[1],
    )
    sys.stdout.flush()

    return train_sub, val_sub, train_split, test_split

def run_tuning(algo: str, train_returns: np.ndarray, val_returns: np.ndarray,
               n_trials: int = 35, use_graph: bool = False) -> optuna.Study:

    study = optuna.create_study(
        study_name=f"{algo}_portfolio_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if algo == "PPO":
        def objective(trial):
            return objective_ppo(trial, train_returns, val_returns, use_graph)
    else:
        def objective(trial):
            return objective_dqn(trial, train_returns, val_returns, use_graph)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def save_results(study: optuna.Study, algo: str):

    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    best = study.best_trial

    results = {
        "algorithm": algo,
        "best_sharpe": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
        "param_importance": _param_importance(study),
        "search_ranges": {},
        "literature_justification": {},
    }

    if algo == "PPO":
        results["search_ranges"] = {
            "learning_rate": "[1e-5, 5e-4] log-scale",
            "gamma": "[0.95, 0.999] via 1-suggest_float",
            "clip_range": "[0.1, 0.2, 0.3] categorical",
            "n_steps": "[256, 2048] powers of 2",
            "batch_size": "[32, 128] powers of 2, capped at n_steps",
            "n_epochs": "[3, 5, 10] categorical",
            "ent_coef": "[1e-4, 0.05] log-scale",
            "gae_lambda": "[0.9, 0.99] via 1-suggest_float",
            "vf_coef": "[0.25, 1.0] uniform",
            "max_grad_norm": "[0.3, 1.0] uniform",
            "hidden_size": "[64, 128, 256] categorical",
            "total_timesteps": "[150K, 400K] step=50K",
        }
        results["literature_justification"] = {
            "learning_rate": "FinRL uses 2.5e-4; ElegantRL uses 3e-5. Lower LR prevents overfitting to noisy financial data.",
            "gamma": "Long-horizon portfolio tasks need high discount. FinRL default: 0.99.",
            "clip_range": "Schulman et al. (2017) PPO paper: 0.2. Tighter clipping helps with noisy financial gradients.",
            "n_steps": "FinRL default: 2048. Must capture enough trajectory for advantage estimation.",
            "ent_coef": "FinRL: 0.01. Prevents premature convergence to degenerate policies.",
            "gae_lambda": "Schulman et al. (2016) GAE: 0.95 standard.",
            "total_timesteps": "FinRL: 100K+. MDPI 2024: 10M. Previous 50K was far too low.",
            "vec_normalize": "Always ON — '37 Implementation Details' (ICLR 2022): observation normalization is most impactful PPO trick.",
            "reward_type": "Moody & Saffell (1998 NeurIPS): Sharpe-based reward yields more robust policies.",
        }
    else:
        results["search_ranges"] = {
            "learning_rate": "[1e-5, 5e-4] log-scale",
            "gamma": "[0.95, 0.999]",
            "buffer_size": "[10K, 100K] log-scale",
            "batch_size": "[32, 256] powers of 2",
            "target_update_interval": "[200, 2000] log-scale",
            "exploration_fraction": "[0.1, 0.4] uniform",
            "exploration_final_eps": "[0.01, 0.1] uniform",
            "learning_starts": "[500, 5000] log-scale",
            "hidden_size": "[64, 128, 256] categorical",
            "total_timesteps": "[100K, 400K] step=50K",
        }
        results["literature_justification"] = {
            "learning_rate": "Nature DQN: 1e-4. Arxiv 2411.07585: 1e-4.",
            "gamma": "Long-horizon financial tasks need high gamma. Standard: 0.99.",
            "buffer_size": "FinRL DDPG: 50K. Non-stationary → old transitions stale.",
            "target_update_interval": "Arxiv 2411.07585: 1000. Financial episodes shorter than Atari.",
            "exploration_fraction": "Extended exploration (0.2-0.3) prevents premature convergence.",
            "total_timesteps": "50K far too low for 100-asset portfolio.",
        }

    out_path = TUNING_DIR / f"{algo.lower()}_best_hyperparams.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved best %s hyperparameters to %s", algo, out_path)

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
            "state": str(t.state),
        })
    trials_path = TUNING_DIR / f"{algo.lower()}_all_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2, default=str)
    logger.info("Saved all %d trials to %s", len(trials_data), trials_path)

    return results

def main():
    parser = argparse.ArgumentParser(description="RL Hyperparameter Tuning with Optuna")
    parser.add_argument("--algo", type=str, default="both", choices=["PPO", "DQN", "both"])
    parser.add_argument("--n_trials", type=int, default=35)
    parser.add_argument("--n_assets", type=int, default=100)
    parser.add_argument("--tune_assets", type=int, default=None,
                        help="Subsample to N assets for faster tuning")
    parser.add_argument("--use_graph", action="store_true",
                        help="Include graph/TDA features (slower)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_returns, val_returns, _, _ = prepare_data(
        n_assets=args.n_assets, tune_assets=args.tune_assets,
        use_graph=args.use_graph,
    )

    algos = ["PPO", "DQN"] if args.algo == "both" else [args.algo]

    for algo in algos:
        logger.info("=" * 60)
        logger.info("Starting %s hyperparameter tuning (%d trials)", algo, args.n_trials)
        logger.info("=" * 60)
        sys.stdout.flush()

        t0 = time.time()
        study = run_tuning(
            algo, train_returns, val_returns,
            n_trials=args.n_trials, use_graph=args.use_graph,
        )
        elapsed = time.time() - t0

        results = save_results(study, algo)

        print(f"\n{'='*60}")
        print(f"{algo} TUNING COMPLETE — {elapsed/60:.1f} minutes")
        print(f"{'='*60}")
        print(f"Best Sharpe: {study.best_value:.4f}")
        print(f"Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        if results.get("param_importance"):
            print(f"\nParameter importance:")
            for k, v in sorted(results["param_importance"].items(),
                               key=lambda x: x[1], reverse=True):
                print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
