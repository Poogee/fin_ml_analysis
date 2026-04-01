from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.pipeline import DataPipeline, DataSplit
from src.data.sentiment import SentimentPipeline
from src.evaluation.backtester import (
    BacktestResult,
    run_walk_forward,
    datasplit_to_dict,
    _get_universe_mask,
)
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import (
    _BaseRLModel,
    _build_price_features,
    _build_graph_tda_features,
    _build_sentiment_features,
)
from src.rl.environment import LongShortPortfolioEnv
from src.rl.train_full_pipeline import (
    _build_additional_features,
    filter_splits,
    RESULTS_DIR as FULL_RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "long_short"
MODELS_DIR = PROJECT_ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FullPipelineLongShortPPO(_BaseRLModel):

    def __init__(
        self,
        strategy: str = "130_30",
        max_long_weight: float = 0.10,
        max_short_weight: float = 0.05,
        annual_borrow_rate: float = 0.005,
        short_tc_premium_bps: float = 5.0,
        use_graph: bool = True,
        use_sentiment: bool = True,
        use_additional: bool = True,
        timesteps: int = 1_000_000,
        reward_type: str = "composite",
        lookback: int = 10,
        max_weight: float = 0.15,
        seed: int = 42,
        net_arch: list[int] | None = None,
        device: str = DEVICE,
        lambda_turnover: float = 1.0,
    ):
        super().__init__(
            algorithm="PPO",
            total_timesteps=timesteps,
            reward_type=reward_type,
            use_vec_normalize=True,
            lookback=lookback,
            max_weight=max_weight,
            seed=seed,
            n_seeds=1,
            net_arch=net_arch or [256, 256],
            lambda_turnover=lambda_turnover,
        )
        self.strategy = strategy
        self.max_long_weight = max_long_weight
        self.max_short_weight = max_short_weight
        self.annual_borrow_rate = annual_borrow_rate
        self.short_tc_premium_bps = short_tc_premium_bps
        self._use_graph = use_graph
        self._use_sentiment = use_sentiment
        self._use_additional = use_additional
        self._device = device

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
            strategy=self.strategy,
            max_long_weight=self.max_long_weight,
            max_short_weight=self.max_short_weight,
            annual_borrow_rate=self.annual_borrow_rate,
            short_tc_premium_bps=self.short_tc_premium_bps,
        )
        env = DummyVecEnv([lambda: LongShortPortfolioEnv(**env_kwargs)])
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
        features_pa = _build_price_features(returns_arr, self.lookback)
        features_global = None

        if config["use_graph"]:
            logger.info("Computing graph/TDA features...")
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
            if sent_global is not None:
                features_global = (
                    np.concatenate([features_global, sent_global], axis=1)
                    if features_global is not None else sent_global
                )

        if self._use_additional:
            logger.info("Computing additional features...")
            t0 = time.time()
            add_pa, add_gl = _build_additional_features(returns_arr)
            logger.info("Additional features computed in %.1fs", time.time() - t0)
            features_pa = np.concatenate([features_pa, add_pa], axis=2)
            features_global = (
                np.concatenate([features_global, add_gl], axis=1)
                if features_global is not None else add_gl
            )

        logger.info(
            "Feature dims — per-asset: %d, global: %s",
            features_pa.shape[2],
            features_global.shape[1] if features_global is not None else 0,
        )

        env = self._make_env(returns_arr, features_pa, features_global)
        agent = self._create_agent(env, self.seed)

        logger.info(
            "Training LongShort PPO (strategy=%s, graph=%s, sent=%s, device=%s) "
            "for %d steps on %d assets...",
            self.strategy, self._use_graph, self._use_sentiment,
            self._device, self.total_timesteps, N,
        )
        agent.learn(total_timesteps=self.total_timesteps)

        self._agent = agent
        if isinstance(env, VecNormalize):
            self._vec_normalize = env
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

        features_pa = _build_price_features(returns_arr, self.lookback)
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
                features_global = (
                    np.concatenate([features_global, sent_global], axis=1)
                    if features_global is not None else sent_global
                )

        if self._use_additional:
            add_pa, add_gl = _build_additional_features(returns_arr)
            features_pa = np.concatenate([features_pa, add_pa], axis=2)
            features_global = (
                np.concatenate([features_global, add_gl], axis=1)
                if features_global is not None else add_gl
            )

        env_kwargs = dict(
            returns=returns_arr,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            max_weight=self.max_weight,
            strategy=self.strategy,
            max_long_weight=self.max_long_weight,
            max_short_weight=self.max_short_weight,
            annual_borrow_rate=self.annual_borrow_rate,
            short_tc_premium_bps=self.short_tc_premium_bps,
        )
        tmp_env = LongShortPortfolioEnv(**env_kwargs)
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

def run_walk_forward_long_short(
    model,
    train_split: DataSplit,
    test_split: DataSplit,
    model_name: str = "Model",
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    short_tc_premium_bps: float = 5.0,
    annual_borrow_rate: float = 0.005,
    rebalance_frequency: int = 5,
    max_lookback: int = 504,
) -> BacktestResult:

    train_data = datasplit_to_dict(train_split)
    model.fit(train_data)

    test_returns = test_split.returns
    test_presence = test_split.presence
    dates = test_returns.index
    n_assets = test_returns.shape[1]
    columns = list(test_returns.columns)

    universe_schedule = None
    if test_split.universe_schedule:
        universe_schedule = {
            **train_split.universe_schedule,
            **test_split.universe_schedule,
        }

    tc_bps = transaction_cost_bps + slippage_bps
    tc_rate = tc_bps / 10000
    short_tc_rate = (tc_bps + short_tc_premium_bps) / 10000
    daily_borrow = annual_borrow_rate / 252

    all_returns = []
    current_weights = np.zeros(n_assets)

    full_returns = pd.concat([train_split.returns, test_split.returns])
    full_prices = pd.concat([train_split.prices, test_split.prices])
    full_presence = pd.concat([train_split.presence, test_split.presence])
    train_len = len(train_split.returns)

    test_returns_np = test_returns.values
    test_presence_np = test_presence.values

    for i in range(len(dates)):
        if i % rebalance_frequency == 0:
            end_idx = train_len + i + 1
            start_idx = max(0, end_idx - max_lookback)
            current_data = {
                "returns": full_returns.iloc[start_idx:end_idx],
                "prices": full_prices.iloc[start_idx:end_idx],
                "presence_mask": full_presence.iloc[start_idx:end_idx],
            }
            new_weights = model.predict_weights(current_data)

            prow = test_presence_np[i]
            mask = prow == 1.0
            new_weights[~mask] = 0.0

            if universe_schedule:
                univ_mask = _get_universe_mask(dates[i], universe_schedule, columns)
                new_weights[~univ_mask] = 0.0

            long_sum = new_weights[new_weights > 0].sum()
            short_sum = abs(new_weights[new_weights < 0].sum())
            net = long_sum - short_sum
            if net > 0 and abs(net - 1.0) > 0.01:

                new_weights = new_weights / net

            weight_changes = new_weights - current_weights
            long_turnover = np.abs(weight_changes[weight_changes > 0]).sum()
            short_turnover = np.abs(weight_changes[weight_changes < 0]).sum()
            tc = long_turnover * tc_rate + short_turnover * short_tc_rate
            current_weights = new_weights
        else:
            tc = 0.0

        short_exposure = abs(np.minimum(current_weights, 0).sum())
        borrow_cost = short_exposure * daily_borrow

        port_ret = np.nansum(current_weights * test_returns_np[i]) - tc - borrow_cost
        all_returns.append(port_ret)

    returns_series = pd.Series(all_returns, index=dates, name="return")
    equity = (1 + returns_series).cumprod()

    return BacktestResult(
        model_name=model_name,
        returns=returns_series,
        equity_curve=equity,
    )

def apply_vol_targeting(returns: pd.Series, target_vol: float = 0.15,
                        vol_window: int = 60) -> pd.Series:

    realized_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    realized_vol = realized_vol.replace(0, np.nan).ffill().bfill()
    scale = (target_vol / realized_vol).clip(0.5, 2.0)
    return returns * scale

def train_eval_ls(model, name, train_split, test_split, save_model=True):

    logger.info("=" * 70)
    logger.info("TRAINING: %s", name)
    logger.info("=" * 70)
    t0 = time.time()

    result = run_walk_forward_long_short(
        model=model,
        train_split=train_split,
        test_split=test_split,
        model_name=name,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        short_tc_premium_bps=5.0,
        annual_borrow_rate=0.005,
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

    logger.info("Loading data (n_assets=100)...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets | Test: %d days",
                train_split.n_days, n, test_split.n_days)

    all_results = {}
    ppo_device = "cpu"

    logger.info("\n" + "▸" * 30 + " 130/30 LONG/SHORT 1M " + "◂" * 30)
    model_130_30 = FullPipelineLongShortPPO(
        strategy="130_30",
        max_long_weight=0.10,
        max_short_weight=0.05,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        lambda_turnover=1.0,
        device=ppo_device,
    )
    result_130, m = train_eval_ls(model_130_30, "LS_130_30_1M", train_split, test_split)
    all_results["LS_130_30_1M"] = m

    logger.info("\n" + "▸" * 30 + " 130/30 + VOL TARGET " + "◂" * 30)
    vt_returns = apply_vol_targeting(result_130.returns, target_vol=0.15)
    vt_equity = (1 + vt_returns).cumprod()
    vt_result = BacktestResult(
        model_name="LS_130_30_VolTarget",
        returns=vt_returns,
        equity_curve=vt_equity,
    )
    vt_metrics = PortfolioMetrics.compute_all(vt_returns)
    all_results["LS_130_30_VolTarget"] = vt_metrics
    vt_returns.to_csv(RESULTS_DIR / "LS_130_30_VolTarget_returns.csv")

    logger.info(
        "LS_130_30_VolTarget: Sharpe=%.3f | Ret=%.2f%% | DD=%.2f%%",
        vt_metrics["Sharpe Ratio"],
        vt_metrics["Cumulative Return"] * 100,
        vt_metrics["Max Drawdown"] * 100,
    )

    logger.info("\n" + "▸" * 30 + " 130/30 SHARPE REWARD " + "◂" * 30)
    model_sharpe = FullPipelineLongShortPPO(
        strategy="130_30",
        max_long_weight=0.10,
        max_short_weight=0.05,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="sharpe",
        lookback=10,
        max_weight=0.15,
        lambda_turnover=1.0,
        device=ppo_device,
    )
    result_sharpe, m = train_eval_ls(model_sharpe, "LS_130_30_Sharpe", train_split, test_split)
    all_results["LS_130_30_Sharpe"] = m

    vt_sharpe = apply_vol_targeting(result_sharpe.returns, target_vol=0.15)
    vt_sharpe_metrics = PortfolioMetrics.compute_all(vt_sharpe)
    all_results["LS_130_30_Sharpe_VT"] = vt_sharpe_metrics
    vt_sharpe.to_csv(RESULTS_DIR / "LS_130_30_Sharpe_VT_returns.csv")
    logger.info(
        "LS_130_30_Sharpe_VT: Sharpe=%.3f | Ret=%.2f%%",
        vt_sharpe_metrics["Sharpe Ratio"],
        vt_sharpe_metrics["Cumulative Return"] * 100,
    )

    logger.info("\n" + "▸" * 30 + " DOLLAR NEUTRAL " + "◂" * 30)
    model_dn = FullPipelineLongShortPPO(
        strategy="dollar_neutral",
        max_long_weight=0.10,
        max_short_weight=0.05,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        lambda_turnover=1.0,
        device=ppo_device,
    )
    _, m = train_eval_ls(model_dn, "LS_DollarNeutral", train_split, test_split)
    all_results["LS_DollarNeutral"] = m

    if all_results["LS_130_30_1M"]["Sharpe Ratio"] > 0.4:
        logger.info("\n" + "▸" * 30 + " 130/30 2M STEPS " + "◂" * 30)
        model_2m = FullPipelineLongShortPPO(
            strategy="130_30",
            max_long_weight=0.10,
            max_short_weight=0.05,
            use_graph=True, use_sentiment=True, use_additional=True,
            timesteps=2_000_000,
            reward_type="composite",
            lookback=10,
            max_weight=0.15,
            lambda_turnover=1.0,
            device=ppo_device,
            seed=123,
        )
        result_2m, m = train_eval_ls(model_2m, "LS_130_30_2M", train_split, test_split)
        all_results["LS_130_30_2M"] = m

        vt_2m = apply_vol_targeting(result_2m.returns, target_vol=0.15)
        vt_2m_metrics = PortfolioMetrics.compute_all(vt_2m)
        all_results["LS_130_30_2M_VT"] = vt_2m_metrics
        vt_2m.to_csv(RESULTS_DIR / "LS_130_30_2M_VT_returns.csv")
        logger.info(
            "LS_130_30_2M_VT: Sharpe=%.3f | Ret=%.2f%%",
            vt_2m_metrics["Sharpe Ratio"],
            vt_2m_metrics["Cumulative Return"] * 100,
        )

    MVO_SHARPE = 0.963
    print("\n" + "═" * 110)
    print("LONG/SHORT PORTFOLIO RESULTS")
    print("═" * 110)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | "
          f"{'Calmar':>7s} | {'Sortino':>7s} | {'Ann.Vol':>7s} | vs MVO")
    print("─" * 110)

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]["Sharpe Ratio"], reverse=True)

    for name, m in sorted_results:
        beat = "✓ BEAT" if m["Sharpe Ratio"] >= MVO_SHARPE else "✗"
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return'] * 100:8.2f}% | "
              f"{m['Max Drawdown'] * 100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f} | "
              f"{m['Annualized Volatility'] * 100:6.2f}% | {beat}")

    print("═" * 110)
    print(f"\nMVO target Sharpe: {MVO_SHARPE:.3f}")

    best = sorted_results[0]
    print(f"Best: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")

    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(RESULTS_DIR / "long_short_summary.csv")
    logger.info("Summary saved to %s", RESULTS_DIR / "long_short_summary.csv")

    return all_results

if __name__ == "__main__":
    main()
