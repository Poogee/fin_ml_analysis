from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.data.sentiment import SentimentPipeline
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import (
    _BaseRLModel,
    _build_price_features,
    _build_graph_tda_features,
    _build_sentiment_features,
)
from src.rl.environment import TiltPortfolioEnv, DiscretePortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "full_pipeline"
MODELS_DIR = PROJECT_ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _build_additional_features(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    T, N = returns.shape
    pa = np.zeros((T, N, 6), dtype=np.float32)
    gl = np.zeros((T, 7), dtype=np.float32)

    cumret = np.nancumsum(returns, axis=0)

    for t in range(T):

        if t >= 60:
            pa[t, :, 0] = cumret[t] - cumret[t - 60]

        if t >= 252:
            pa[t, :, 1] = cumret[t] - cumret[t - 252]

        if t >= 5:
            pa[t, :, 2] = np.nanstd(returns[t - 5:t], axis=0) * np.sqrt(252)

        if t >= 60:
            pa[t, :, 3] = np.nanstd(returns[t - 60:t], axis=0) * np.sqrt(252)

        if t >= 20:
            mom20 = cumret[t] - cumret[t - 20]
            order = np.argsort(np.argsort(mom20))
            pa[t, :, 4] = order / max(N - 1, 1)

        if t >= 50:
            window = returns[t - 50:t]
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0)
            std[std < 1e-8] = 1.0
            pa[t, :, 5] = np.clip((returns[t] - mean) / std, -3, 3)

        if t >= 20:
            mom20 = cumret[t] - cumret[t - 20]
            gl[t, 0] = np.mean(mom20 > 0)

        gl[t, 1] = np.nanstd(returns[t])

        mkt = np.nanmean(returns[:t + 1], axis=1)
        if t >= 5:
            gl[t, 2] = mkt[t - 5:t].sum()
        if t >= 20:
            gl[t, 3] = mkt[t - 20:t].sum()
        if t >= 60:
            gl[t, 4] = mkt[t - 60:t].sum()

        if t >= 20:
            gl[t, 5] = np.std(mkt[t - 20:t]) * np.sqrt(252)

    for t in range(60, T, 15):
        window = returns[t - 60:t]
        window = np.nan_to_num(window, nan=0.0)
        corr = np.corrcoef(window.T)
        corr = np.nan_to_num(corr, nan=0.0)
        upper = corr[np.triu_indices(N, k=1)]
        gl[t, 6] = upper.mean()

    for t in range(1, T):
        if gl[t, 6] == 0.0 and t >= 60:
            gl[t, 6] = gl[t - 1, 6]

    return pa, gl

class FullPipelineTiltPPO(_BaseRLModel):

    def __init__(
        self,
        tilt_scale: float = 0.3,
        use_graph: bool = True,
        use_sentiment: bool = True,
        use_additional: bool = True,
        timesteps: int = 1_000_000,
        reward_type: str = "composite",
        lookback: int = 10,
        max_weight: float = 0.15,
        seed: int = 42,
        n_seeds: int = 1,
        net_arch: list[int] | None = None,
        device: str = DEVICE,
    ):
        super().__init__(
            algorithm="PPO",
            total_timesteps=timesteps,
            reward_type=reward_type,
            use_vec_normalize=True,
            lookback=lookback,
            max_weight=max_weight,
            seed=seed,
            n_seeds=n_seeds,
            net_arch=net_arch or [256, 256],
        )
        self.tilt_scale = tilt_scale
        self._use_graph = use_graph
        self._use_sentiment = use_sentiment
        self._use_additional = use_additional
        self._device = device

        self._additional_pa_cache = None
        self._additional_gl_cache = None

    def _get_feature_config(self) -> dict:
        return {
            "use_graph": self._use_graph,
            "use_sentiment": self._use_sentiment,
        }

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
            tilt_scale=self.tilt_scale,
        )
        env = DummyVecEnv([lambda: TiltPortfolioEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env

    def _create_agent(self, env, seed_val):

        pk = dict(
            net_arch=dict(pi=list(self.net_arch), vf=list(self.net_arch)),
            activation_fn=torch.nn.Tanh,
        )
        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.982,
            gae_lambda=0.976,
            clip_range=0.3,
            ent_coef=8e-4,
            vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=pk,
            verbose=0,
            seed=seed_val,
            device=self._device,
        )

    def fit(self, train_data: dict) -> None:

        returns = train_data["returns"]
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.values
        else:
            returns_arr = returns
        returns_arr = np.nan_to_num(returns_arr, nan=0.0).astype(np.float32)

        T, N = returns_arr.shape
        self._n_assets = N
        config = self._get_feature_config()

        logger.info("Building price features...")
        price_feats = _build_price_features(returns_arr, self.lookback)
        features_pa = price_feats
        features_global = None

        if config["use_graph"]:
            logger.info("Computing graph/TDA features (spectral + persistent homology)...")
            t0 = time.time()
            graph_pa, graph_global = _build_graph_tda_features(
                returns_arr,
                corr_window=self.corr_window,
                graph_k=self.graph_k,
                diffusion_times=self.diffusion_times,
                n_clusters=self.n_clusters,
            )
            logger.info("Graph/TDA features computed in %.1fs", time.time() - t0)
            features_pa = np.concatenate([features_pa, graph_pa], axis=2)
            features_global = graph_global

        if config["use_sentiment"]:
            logger.info("Building sentiment features...")
            daily_sent = self._load_sentiment()
            sent_pa, sent_global = _build_sentiment_features(
                config, returns_arr, train_data, daily_sent,
            )
            if sent_pa is not None:
                features_pa = np.concatenate([features_pa, sent_pa], axis=2)
                logger.info("Sentiment per-asset: %d dims", sent_pa.shape[2])
            if sent_global is not None:
                if features_global is not None:
                    features_global = np.concatenate([features_global, sent_global], axis=1)
                else:
                    features_global = sent_global
                logger.info("Sentiment global: %d dims", sent_global.shape[1])

        if self._use_additional:
            logger.info("Computing additional features (momentum, vol, breadth)...")
            t0 = time.time()
            add_pa, add_gl = _build_additional_features(returns_arr)
            logger.info("Additional features computed in %.1fs", time.time() - t0)
            features_pa = np.concatenate([features_pa, add_pa], axis=2)
            if features_global is not None:
                features_global = np.concatenate([features_global, add_gl], axis=1)
            else:
                features_global = add_gl

        logger.info(
            "Feature dimensions — per-asset: %d, global: %s, obs_dim ≈ %d",
            features_pa.shape[2],
            features_global.shape[1] if features_global is not None else 0,
            N * self.lookback * (1 + features_pa.shape[2]) + N
            + (features_global.shape[1] if features_global is not None else 0),
        )

        best_agent = None
        best_env = None
        best_score = -np.inf

        seeds = [self.seed + i * 1000 for i in range(self.n_seeds)]

        for i, seed in enumerate(seeds):
            if self.n_seeds > 1:
                logger.info("Training seed %d/%d (seed=%d)...", i + 1, self.n_seeds, seed)

            env = self._make_env(returns_arr, features_pa, features_global)
            agent = self._create_agent(env, seed)

            logger.info(
                "Training %s (graph=%s, sent=%s, add=%s, device=%s) for %d steps on %d assets...",
                self.__class__.__name__,
                self._use_graph, self._use_sentiment, self._use_additional,
                self._device, self.total_timesteps, N,
            )
            agent.learn(total_timesteps=self.total_timesteps)

            if self.n_seeds > 1:
                eval_env = self._make_env(returns_arr, features_pa, features_global)
                if isinstance(eval_env, VecNormalize) and isinstance(env, VecNormalize):
                    eval_env.obs_rms = env.obs_rms
                    eval_env.ret_rms = env.ret_rms
                    eval_env.training = False
                    eval_env.norm_reward = False
                score = self._evaluate_agent_on_train(agent, eval_env)
                logger.info("  Seed %d score: %.4f", seed, score)
                if score > best_score:
                    best_score = score
                    best_agent = agent
                    best_env = env
            else:
                best_agent = agent
                best_env = env

        self._agent = best_agent
        if isinstance(best_env, VecNormalize):
            self._vec_normalize = best_env
        logger.info("Training complete.")

        if config["use_graph"]:
            from src.features.graph import CorrelationGraph
            from src.features.tda import TDAFeatureExtractor
            self._graph = CorrelationGraph(
                method="knn", k=self.graph_k,
                diffusion_times=self.diffusion_times,
            )
            self._tda = TDAFeatureExtractor(max_homology_dim=1)

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.values
        else:
            returns_arr = returns
        returns_arr = np.nan_to_num(returns_arr, nan=0.0).astype(np.float32)

        n_assets = returns_arr.shape[1]
        if self._agent is None:
            return np.ones(n_assets, dtype=np.float32) / n_assets

        config = self._get_feature_config()

        price_feats = _build_price_features(returns_arr, self.lookback)
        features_pa = price_feats
        features_global = None

        if config["use_graph"]:
            graph_pa, graph_global = _build_graph_tda_features(
                returns_arr,
                corr_window=self.corr_window,
                graph_k=self.graph_k,
                diffusion_times=self.diffusion_times,
                n_clusters=self.n_clusters,
                recompute_freq=max(1, len(returns_arr) // 10),
            )
            features_pa = np.concatenate([features_pa, graph_pa], axis=2)
            features_global = graph_global

        if config["use_sentiment"]:
            daily_sent = self._load_sentiment()
            sent_pa, sent_global = _build_sentiment_features(
                config, returns_arr, current_data, daily_sent,
            )
            if sent_pa is not None:
                features_pa = np.concatenate([features_pa, sent_pa], axis=2)
            if sent_global is not None:
                if features_global is not None:
                    features_global = np.concatenate([features_global, sent_global], axis=1)
                else:
                    features_global = sent_global

        if self._use_additional:
            add_pa, add_gl = _build_additional_features(returns_arr)
            features_pa = np.concatenate([features_pa, add_pa], axis=2)
            if features_global is not None:
                features_global = np.concatenate([features_global, add_gl], axis=1)
            else:
                features_global = add_gl

        env_kwargs = dict(
            returns=returns_arr,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            max_weight=self.max_weight,
            tilt_scale=self.tilt_scale,
        )
        tmp_env = TiltPortfolioEnv(**env_kwargs)
        tmp_env._step = len(returns_arr) - 1
        tmp_env._weights = np.zeros(n_assets, dtype=np.float32)
        obs = tmp_env._get_obs()

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._agent.predict(obs, deterministic=True)
        return tmp_env._normalize_weights(action)

    def save_model(self, path: Path, name: str) -> None:

        path.mkdir(parents=True, exist_ok=True)
        if self._agent is not None:
            self._agent.save(str(path / f"{name}_agent"))
        if self._vec_normalize is not None:
            self._vec_normalize.save(str(path / f"{name}_vecnorm.pkl"))
        logger.info("Model saved to %s", path / name)

class TunedDQN(_BaseRLModel):

    def __init__(self, seed: int = 42, timesteps: int = 400_000,
                 lookback: int = 10, max_weight: float = 0.15):
        super().__init__(
            algorithm="DQN",
            total_timesteps=timesteps,
            reward_type="sharpe",
            use_vec_normalize=False,
            lookback=lookback,
            max_weight=max_weight,
            seed=seed,
            net_arch=[64, 64],
        )

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed):
        pk = dict(net_arch=[64, 64], activation_fn=torch.nn.ReLU)
        return DQN(
            "MlpPolicy", env,
            learning_rate=2.1e-5,
            buffer_size=17_000,
            learning_starts=1636,
            batch_size=64,
            gamma=0.996,
            target_update_interval=680,
            exploration_fraction=0.30,
            exploration_final_eps=0.073,
            policy_kwargs=pk,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

    def save_model(self, path: Path, name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._agent is not None:
            self._agent.save(str(path / f"{name}_agent"))
        logger.info("DQN model saved to %s", path / name)

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

def train_eval(model, name, train_split, test_split, save_model=True):

    logger.info("=" * 70)
    logger.info("TRAINING: %s", name)
    logger.info("=" * 70)
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
        metrics["Cumulative Return"] * 100,
        metrics["Max Drawdown"] * 100, metrics["Calmar Ratio"],
        metrics["Sortino Ratio"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace(" ", "_")
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)

    if save_model and hasattr(model, "save_model"):
        model.save_model(MODELS_DIR, safe)

    return result, metrics

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)
    if DEVICE == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.0f MB free", torch.cuda.mem_get_info()[0] / 1e6)

    logger.info("Loading data (n_assets=100)...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets | Test: %d days",
                train_split.n_days, n, test_split.n_days)

    all_results = {}
    TARGETS = {"MVO": 0.963, "LSTM": 0.908, "SP500": 0.72}

    ppo_device = "cpu"

    logger.info("\n" + "▸" * 30 + " PHASE 1: FULL PIPELINE " + "◂" * 30)
    model_full = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        device=ppo_device,
    )
    _, m = train_eval(model_full, "Full_Pipeline_1M", train_split, test_split)
    all_results["Full_Pipeline_1M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 2: NO SENTIMENT " + "◂" * 30)
    model_no_sent = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=True, use_sentiment=False, use_additional=True,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        device=ppo_device,
    )
    _, m = train_eval(model_no_sent, "No_Sentiment_1M", train_split, test_split)
    all_results["No_Sentiment_1M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 3: NO GRAPH " + "◂" * 30)
    model_no_graph = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=False, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        device=ppo_device,
    )
    _, m = train_eval(model_no_graph, "No_Graph_1M", train_split, test_split)
    all_results["No_Graph_1M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 4: PRICE ONLY " + "◂" * 30)
    model_price = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=False, use_sentiment=False, use_additional=False,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        device=ppo_device,
    )
    _, m = train_eval(model_price, "Price_Only_1M", train_split, test_split)
    all_results["Price_Only_1M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 5: DQN (3 seeds) " + "◂" * 30)
    dqn_sharpes = []
    for seed in [42, 1042, 2042]:
        name = f"DQN_seed{seed}_400K"
        model_dqn = TunedDQN(seed=seed, timesteps=400_000)
        _, m = train_eval(model_dqn, name, train_split, test_split)
        all_results[name] = m
        dqn_sharpes.append(m["Sharpe Ratio"])

    logger.info(
        "DQN Sharpe: mean=%.3f, std=%.3f (seeds: %s)",
        np.mean(dqn_sharpes), np.std(dqn_sharpes),
        [f"{s:.3f}" for s in dqn_sharpes],
    )

    full_sharpe = all_results["Full_Pipeline_1M"]["Sharpe Ratio"]
    if full_sharpe > 0.5:
        logger.info("\n" + "▸" * 30 + " PHASE 6: FULL 2M STEPS " + "◂" * 30)
        model_full_2m = FullPipelineTiltPPO(
            tilt_scale=0.3,
            use_graph=True, use_sentiment=True, use_additional=True,
            timesteps=2_000_000,
            reward_type="composite",
            lookback=10,
            max_weight=0.15,
            device=ppo_device,
        )
        _, m = train_eval(model_full_2m, "Full_Pipeline_2M", train_split, test_split)
        all_results["Full_Pipeline_2M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 7: FULL SHARPE REWARD " + "◂" * 30)
    model_full_sharpe = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="sharpe",
        lookback=10,
        max_weight=0.15,
        device=ppo_device,
    )
    _, m = train_eval(model_full_sharpe, "Full_Sharpe_1M", train_split, test_split)
    all_results["Full_Sharpe_1M"] = m

    print("\n" + "═" * 110)
    print("FULL PIPELINE ABLATION STUDY RESULTS")
    print("═" * 110)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | "
          f"{'Calmar':>7s} | {'Sortino':>7s} | {'Ann.Vol':>7s}")
    print("─" * 110)

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["Sharpe Ratio"], reverse=True)

    for name, m in sorted_results:
        flags = []
        for target_name, target_val in TARGETS.items():
            if m["Sharpe Ratio"] >= target_val:
                flags.append(f">{target_name}")
        flag_str = " " + ",".join(flags) if flags else ""
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return'] * 100:8.2f}% | "
              f"{m['Max Drawdown'] * 100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f} | "
              f"{m['Annualized Volatility'] * 100:6.2f}%{flag_str}")

    print("═" * 110)

    best = sorted_results[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")

    ablation_names = {
        "Full_Pipeline_1M": "FULL (graph+TDA+sent+add)",
        "No_Sentiment_1M": "No Sentiment (graph+TDA+add)",
        "No_Graph_1M": "No Graph (sent+add)",
        "Price_Only_1M": "Price Only (baseline)",
    }
    print("\nABLATION ANALYSIS:")
    print("─" * 60)
    base_sharpe = all_results.get("Price_Only_1M", {}).get("Sharpe Ratio", 0)
    for name, desc in ablation_names.items():
        if name in all_results:
            s = all_results[name]["Sharpe Ratio"]
            delta = s - base_sharpe
            print(f"  {desc:<40s}: Sharpe {s:.3f} (Δ={delta:+.3f})")

    if dqn_sharpes:
        print(f"\nDQN (3 seeds): {np.mean(dqn_sharpes):.3f} ± {np.std(dqn_sharpes):.3f}")

    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(RESULTS_DIR / "full_pipeline_summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "full_pipeline_summary.csv")

if __name__ == "__main__":
    main()
