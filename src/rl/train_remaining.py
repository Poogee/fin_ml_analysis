from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "full_pipeline"

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

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    if hasattr(model, "_agent") and model._agent is not None:
        model._agent.save(str(models_dir / f"{safe}_agent"))
        logger.info("Model saved to %s", models_dir / safe)
    if hasattr(model, "_vec_normalize") and model._vec_normalize is not None:
        with open(models_dir / f"{safe}_vecnorm.pkl", "wb") as f:
            pickle.dump(model._vec_normalize, f)

    return result, metrics

def main():

    from src.rl.train_full_pipeline import FullPipelineTiltPPO

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=100)...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets", train_split.n_days, n)

    all_results = {}

    logger.info("\n" + "▸" * 30 + " PHASE 6: FULL 2M STEPS (CPU) " + "◂" * 30)
    model_2m = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=2_000_000,
        reward_type="composite",
        lookback=10,
        max_weight=0.15,
        device="cpu",
    )
    _, m = train_eval(model_2m, "Full_Pipeline_2M", train_split, test_split)
    all_results["Full_Pipeline_2M"] = m

    logger.info("\n" + "▸" * 30 + " PHASE 7: FULL SHARPE REWARD (CPU) " + "◂" * 30)
    model_sharpe = FullPipelineTiltPPO(
        tilt_scale=0.3,
        use_graph=True, use_sentiment=True, use_additional=True,
        timesteps=1_000_000,
        reward_type="sharpe",
        lookback=10,
        max_weight=0.15,
        device="cpu",
    )
    _, m = train_eval(model_sharpe, "Full_Sharpe_1M", train_split, test_split)
    all_results["Full_Sharpe_1M"] = m

    print("\n" + "=" * 80)
    print("REMAINING PHASES RESULTS")
    print("=" * 80)
    for name, m in all_results.items():
        print(f"  {name:<30s} | Sharpe={m['Sharpe Ratio']:.3f} | "
              f"Ret={m['Cumulative Return']*100:.2f}% | "
              f"DD={m['Max Drawdown']*100:.2f}% | "
              f"Calmar={m['Calmar Ratio']:.3f} | "
              f"Sortino={m['Sortino Ratio']:.3f}")
    print("=" * 80)

    existing = {}
    summary_path = RESULTS_DIR / "full_pipeline_summary.csv"

    for csv in RESULTS_DIR.glob("*_returns.csv"):
        model_name = csv.stem.replace("_returns", "")
        if model_name not in all_results:
            try:
                rets = pd.read_csv(csv, index_col=0, parse_dates=True).squeeze()
                existing[model_name] = PortfolioMetrics.compute_all(rets)
            except Exception:
                pass

    all_combined = {**existing, **all_results}
    summary_df = pd.DataFrame(all_combined).T
    summary_df.to_csv(summary_path)
    logger.info("Full summary saved to %s (%d models)", summary_path, len(all_combined))

if __name__ == "__main__":
    main()
