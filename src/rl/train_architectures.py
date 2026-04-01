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
from src.rl.extractors import (
    CNN1DExtractor,
    CNNAttentionExtractor,
    ASPPExtractor,
    GATExtractor,
    build_knn_adjacency,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "architectures"

def filter_splits_to_common_universe(
    train_split: DataSplit, test_split: DataSplit,
) -> tuple[DataSplit, DataSplit]:

    all_universe = sorted(set(train_split.all_assets) | set(test_split.all_assets))
    common_cols = set(train_split.returns.columns) & set(test_split.returns.columns)
    available = sorted(a for a in all_universe if a in common_cols)
    if not available:
        return train_split, test_split

    logger.info(
        "Filtering to %d common universe assets (from %d train, %d test cols)",
        len(available), len(train_split.returns.columns),
        len(test_split.returns.columns),
    )

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

class ArchitectureModel(_BaseRLModel):

    def __init__(
        self,
        arch: str = "mlp",
        extractor_kwargs: dict | None = None,
        **kwargs,
    ):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 1_000_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)
        self.arch = arch
        self._extractor_kwargs = extractor_kwargs or {}

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

    def _create_agent(self, env, seed: int):

        if self.arch == "mlp":

            net_arch = self.net_arch if self.net_arch else [256, 256]
            policy_kwargs = dict(
                net_arch=net_arch,
                activation_fn=torch.nn.Tanh,
            )
        else:

            extractor_class, extractor_kwargs = self._get_extractor_config(env)
            policy_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=extractor_kwargs,
                net_arch=dict(pi=[128], vf=[128]),
                activation_fn=torch.nn.ReLU,
                share_features_extractor=False,
            )

        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4,
            n_steps=512,
            batch_size=32,
            n_epochs=10,
            gamma=0.982,
            gae_lambda=0.976,
            clip_range=0.3,
            ent_coef=8e-4,
            vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

    def _get_extractor_config(self, env) -> tuple[type, dict]:

        if isinstance(env, VecNormalize):
            obs_space = env.observation_space
        else:
            obs_space = env.observation_space

        n_assets = self._n_assets
        lookback = self.lookback

        obs_dim = obs_space.shape[0]

        raw_env = env
        while hasattr(raw_env, 'venv'):
            raw_env = raw_env.venv
        if hasattr(raw_env, 'envs'):
            base_env = raw_env.envs[0]
        else:
            base_env = raw_env

        n_pa_features = getattr(base_env, 'n_pa_features', 13)
        n_global_features = getattr(base_env, 'n_global_features', 23)

        common_kwargs = dict(
            n_assets=n_assets,
            lookback=lookback,
            n_pa_features=n_pa_features,
            n_global_features=n_global_features,
            features_dim=256,
        )
        common_kwargs.update(self._extractor_kwargs)

        if self.arch == "cnn":
            return CNN1DExtractor, common_kwargs
        elif self.arch == "aspp":
            common_kwargs.setdefault("aspp_out_channels", 64)
            common_kwargs.setdefault("atrous_rates", (1, 2, 4, 8))
            return ASPPExtractor, common_kwargs
        elif self.arch == "cnn_attn":
            common_kwargs.setdefault("cnn_channels", 64)
            common_kwargs.setdefault("n_heads", 5)
            return CNNAttentionExtractor, common_kwargs
        elif self.arch == "gat":

            adj = self._build_adj_matrix(n_assets)
            common_kwargs["adj_matrix"] = adj
            return GATExtractor, common_kwargs
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

    def _build_adj_matrix(self, n_assets: int) -> torch.Tensor:

        if hasattr(self, '_train_returns') and self._train_returns is not None:

            returns = self._train_returns[-120:]
            return build_knn_adjacency(
                returns=torch.tensor(returns, dtype=torch.float32),
                k=min(10, n_assets - 1),
            )
        return torch.ones(n_assets, n_assets)

    def fit(self, train_data: dict) -> None:

        returns = train_data["returns"]
        if isinstance(returns, pd.DataFrame):
            self._train_returns = returns.values.astype(np.float32)
        else:
            self._train_returns = np.array(returns, dtype=np.float32)
        super().fit(train_data)

def train_eval(
    model, name: str,
    train_split: DataSplit, test_split: DataSplit,
    tc: float = 10.0, slip: float = 5.0, rebal: int = 5,
) -> tuple[BacktestResult, dict]:

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

    logger.info(
        "%s: %.0fs | Sharpe=%.3f | Ret=%.2f%% | DD=%.2f%% | Calmar=%.3f",
        name, elapsed,
        metrics["Sharpe Ratio"],
        metrics["Cumulative Return"] * 100,
        metrics["Max Drawdown"] * 100,
        metrics["Calmar Ratio"],
    )

    safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
    with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
        pickle.dump(result, f)

    return result, metrics

def main():
    parser = argparse.ArgumentParser(description="Train PPO with different NN architectures")
    parser.add_argument(
        "--arch", type=str, default="all",
        choices=["all", "mlp", "cnn", "aspp", "cnn_attn", "gat"],
        help="Which architecture to train (default: all)",
    )
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--n_assets", type=int, default=100, help="Number of assets")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback window")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=%d)...", args.n_assets)
    pipeline = DataPipeline(n_assets=args.n_assets)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits_to_common_universe(train_split, test_split)

    logger.info(
        "Train: %d days, %d assets | Test: %d days, %d assets",
        train_split.n_days, train_split.n_assets,
        test_split.n_days, test_split.n_assets,
    )

    common_kwargs = dict(
        total_timesteps=args.steps,
        lookback=args.lookback,
        max_weight=0.05,
        reward_type="composite",
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        seed=args.seed,
        use_vec_normalize=True,
        lambda_dd=2.0,
        lambda_turnover=0.5,
        dd_threshold=0.05,
        corr_window=60,
        graph_k=10,
        n_clusters=4,
    )

    architectures = {
        "mlp": "PPO_MLP_baseline",
        "cnn": "PPO_CNN1D",
        "aspp": "PPO_ASPP_DeepLabV3",
        "cnn_attn": "PPO_CNN_Attention",
        "gat": "PPO_GAT",
    }

    if args.arch != "all":
        architectures = {args.arch: architectures[args.arch]}

    all_results = {}

    for arch_key, name in architectures.items():
        logger.info("\n" + "=" * 70)
        logger.info("Architecture: %s (%s)", name, arch_key)
        logger.info("Device: %s", DEVICE)
        logger.info("=" * 70)

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model = ArchitectureModel(arch=arch_key, **common_kwargs)
        _, metrics = train_eval(model, name, train_split, test_split)
        all_results[name] = metrics

        if DEVICE == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info("  GPU peak memory: %.0f MB", peak_mb)

    logger.info("\n" + "=" * 70)
    logger.info("ARCHITECTURE COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-25s %8s %8s %8s %8s",
        "Architecture", "Sharpe", "Return%", "MaxDD%", "Calmar",
    )
    logger.info("-" * 70)

    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True,
    )
    for name, m in sorted_results:
        logger.info(
            "%-25s %8.3f %8.2f %8.2f %8.3f",
            name,
            m["Sharpe Ratio"],
            m["Cumulative Return"] * 100,
            m["Max Drawdown"] * 100,
            m["Calmar Ratio"],
        )

    if len(sorted_results) > 1:
        best_name = sorted_results[0][0]
        best_sharpe = sorted_results[0][1]["Sharpe Ratio"]
        baseline_sharpe = all_results.get(
            "PPO_MLP_baseline", {"Sharpe Ratio": 0},
        )["Sharpe Ratio"]

        if baseline_sharpe > 0:
            improvement = (best_sharpe - baseline_sharpe) / baseline_sharpe * 100
            logger.info(
                "\nBest: %s (Sharpe %.3f, %.1f%% improvement over MLP baseline)",
                best_name, best_sharpe, improvement,
            )
        else:
            logger.info("\nBest: %s (Sharpe %.3f)", best_name, best_sharpe)

    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(RESULTS_DIR / "architecture_comparison.csv")
    logger.info("Results saved to %s", RESULTS_DIR)

if __name__ == "__main__":
    main()
