from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import DataPipeline
from src.evaluation.backtester import run_walk_forward, BacktestResult
from src.evaluation.metrics import PortfolioMetrics
from src.rl.agent import RLFullModel, RLNoGraphModel, RLNoSentimentModel, RLDQNModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def train_and_evaluate(
    model,
    name: str,
    train_split,
    test_split,
    tc_bps: float = 10.0,
    slip_bps: float = 5.0,
    rebalance_freq: int = 5,
) -> BacktestResult:

    logger.info("=" * 50)
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

    logger.info(
        "%s completed in %.1fs",
        name, elapsed,
    )
    logger.info(
        "  Sharpe: %.3f | Cum.Return: %.2f%% | Max DD: %.2f%% | Calmar: %.3f",
        metrics["Sharpe Ratio"],
        metrics["Cumulative Return"] * 100,
        metrics["Max Drawdown"] * 100,
        metrics["Calmar Ratio"],
    )

    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    result.returns.to_csv(RESULTS_DIR / f"{safe_name}_returns.csv")
    with open(RESULTS_DIR / f"{safe_name}_result.pkl", "wb") as f:
        pickle.dump(result, f)
    logger.info("  Saved to %s", RESULTS_DIR / f"{safe_name}_result.pkl")

    return result

def main():
    parser = argparse.ArgumentParser(description="Train RL portfolio agents")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Training timesteps per agent (default: 200K)")
    parser.add_argument("--n_assets", type=int, default=100, help="Number of assets in universe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds to try per model (best is kept)")
    parser.add_argument("--reward_type", type=str, default="composite",
                        choices=["return", "sharpe", "dsr", "log_return", "composite"],
                        help="Reward function type")
    parser.add_argument("--no-dqn", action="store_true", help="Skip DQN training")
    parser.add_argument("--tc_bps", type=float, default=10.0, help="Transaction cost (bps)")
    parser.add_argument("--slip_bps", type=float, default=5.0, help="Slippage (bps)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    logger.info("Building data pipeline (n_assets=%d)...", args.n_assets)
    pipeline = DataPipeline(n_assets=args.n_assets)
    train_split, test_split = pipeline.build()

    logger.info(
        "Train: %s to %s (%d days, %d assets)",
        train_split.start_date.date(), train_split.end_date.date(),
        train_split.n_days, train_split.n_assets,
    )
    logger.info(
        "Test: %s to %s (%d days, %d assets)",
        test_split.start_date.date(), test_split.end_date.date(),
        test_split.n_days, test_split.n_assets,
    )

    common_kwargs = dict(
        total_timesteps=args.timesteps,
        transaction_cost_bps=args.tc_bps,
        slippage_bps=args.slip_bps,
        seed=args.seed,
        n_seeds=args.n_seeds,
        max_weight=0.05,
        reward_type=args.reward_type,
        use_vec_normalize=True,
        net_arch=[256, 256],
    )

    results = {}

    model = RLFullModel(**common_kwargs)
    results["RL Full (PPO+Graph+TDA+NLP)"] = train_and_evaluate(
        model, "RL Full (PPO+Graph+TDA+NLP)",
        train_split, test_split, args.tc_bps, args.slip_bps,
    )

    model = RLNoGraphModel(**common_kwargs)
    results["RL No Graph Features"] = train_and_evaluate(
        model, "RL No Graph Features",
        train_split, test_split, args.tc_bps, args.slip_bps,
    )

    model = RLNoSentimentModel(**common_kwargs)
    results["RL No Sentiment"] = train_and_evaluate(
        model, "RL No Sentiment",
        train_split, test_split, args.tc_bps, args.slip_bps,
    )

    if not args.no_dqn:
        dqn_kwargs = {**common_kwargs, "use_vec_normalize": False}
        model = RLDQNModel(**dqn_kwargs)
        results["RL DQN (Full)"] = train_and_evaluate(
            model, "RL DQN (Full)",
            train_split, test_split, args.tc_bps, args.slip_bps,
        )

    print("\n" + "=" * 70)
    print("RL TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        metrics = PortfolioMetrics.compute_all(result.returns)
        print(f"  {name:35s} | Sharpe: {metrics['Sharpe Ratio']:7.3f} | "
              f"Return: {metrics['Cumulative Return']*100:7.2f}% | "
              f"MaxDD: {metrics['Max Drawdown']*100:7.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
