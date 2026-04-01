import sys
import time
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pipeline import DataPipeline
from src.evaluation.metrics import PortfolioMetrics
from src.evaluation.backtester import BacktestResult, run_walk_forward
from src.evaluation.comparison import ModelComparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def build_all_models() -> dict:

    models = {}

    from src.baselines.equal_weight import EqualWeightModel
    models["Equal Weight"] = EqualWeightModel()

    from src.baselines.risk_parity import RiskParityModel
    models["Risk Parity"] = RiskParityModel(vol_lookback=60)

    from src.baselines.mean_variance import MeanVarianceModel
    models["Mean-Variance (Markowitz)"] = MeanVarianceModel(
        lookback=252, max_weight=0.05, risk_aversion=1.0, shrinkage=0.5
    )

    from src.baselines.xgboost_model import XGBoostModel
    models["XGBoost"] = XGBoostModel(
        lookback=20, top_k=50, max_weight=0.05,
        n_estimators=200, max_depth=4, learning_rate=0.05,
    )

    from src.baselines.lightgbm_model import LightGBMModel
    models["LightGBM"] = LightGBMModel(
        lookback=20, top_k=50, max_weight=0.05,
        n_estimators=200, max_depth=4, learning_rate=0.05,
    )

    from src.baselines.lstm_model import LSTMModel
    models["LSTM"] = LSTMModel(
        seq_len=20, top_k=50, max_weight=0.05,
        hidden_size=64, num_layers=2, epochs=10, batch_size=512,
    )

    from src.baselines.transformer_model import TransformerModel
    models["Transformer"] = TransformerModel(
        seq_len=20, top_k=50, max_weight=0.05,
        d_model=64, nhead=4, num_layers=2, epochs=10, batch_size=512,
    )

    try:
        from src.rl.agent import RLFullModel
        models["RL Full (PPO+Graph+TDA+NLP)"] = RLFullModel()
    except (ImportError, ModuleNotFoundError):
        logger.info("RL Full model not available yet")

    try:
        from src.rl.agent import RLNoGraphModel
        models["RL No Graph Features"] = RLNoGraphModel()
    except (ImportError, ModuleNotFoundError):
        logger.info("RL No-Graph ablation not available yet")

    try:
        from src.rl.agent import RLNoSentimentModel
        models["RL No Sentiment"] = RLNoSentimentModel()
    except (ImportError, ModuleNotFoundError):
        logger.info("RL No-Sentiment ablation not available yet")

    return models

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION — WALK-FORWARD EVALUATION")
    print("=" * 60)

    logger.info("Building data pipeline...")
    pipeline = DataPipeline()
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

    assets = test_split.all_assets
    logger.info("Evaluating on %d assets from test universe", len(assets))

    merged_schedule = {
        **train_split.universe_schedule,
        **test_split.universe_schedule,
    }
    logger.info(
        "Universe schedule: %d rebalance dates (prevents look-ahead bias)",
        len(merged_schedule),
    )

    from src.data.pipeline import DataSplit
    train_aligned = DataSplit(
        prices=train_split.prices[assets],
        returns=train_split.returns[assets],
        log_returns=train_split.log_returns[assets],
        market_caps=train_split.market_caps[assets],
        dividends=train_split.dividends[assets],
        presence=train_split.presence[assets],
        universe_schedule=train_split.universe_schedule,
        start_date=train_split.start_date,
        end_date=train_split.end_date,
        all_assets=assets,
    )
    test_aligned = DataSplit(
        prices=test_split.prices[assets],
        returns=test_split.returns[assets],
        log_returns=test_split.log_returns[assets],
        market_caps=test_split.market_caps[assets],
        dividends=test_split.dividends[assets],
        presence=test_split.presence[assets],
        universe_schedule=test_split.universe_schedule,
        start_date=test_split.start_date,
        end_date=test_split.end_date,
        all_assets=assets,
    )

    models = build_all_models()
    logger.info("Models to evaluate: %s", list(models.keys()))

    all_results = {}
    for name, model in models.items():
        logger.info("=" * 50)
        logger.info("Evaluating: %s", name)
        t0 = time.time()

        try:
            result = run_walk_forward(
                model=model,
                train_split=train_aligned,
                test_split=test_aligned,
                model_name=name,
                transaction_cost_bps=10.0,
                slippage_bps=5.0,
                rebalance_frequency=5,
                universe_schedule=merged_schedule,
            )
            elapsed = time.time() - t0
            metrics = PortfolioMetrics.compute_all(result.returns)
            logger.info(
                "%s done in %.1fs — Sharpe: %.3f, CumRet: %.2f%%, MaxDD: %.2f%%",
                name, elapsed,
                metrics["Sharpe Ratio"],
                metrics["Cumulative Return"] * 100,
                metrics["Max Drawdown"] * 100,
            )
            all_results[name] = result

            safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("+", "")
            result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
            with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            logger.error("FAILED: %s — %s", name, e, exc_info=True)

    if not all_results:
        logger.error("No results produced!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GENERATING COMPARISON OUTPUTS")
    print("=" * 60)

    comparison = ModelComparison(all_results, output_dir=str(RESULTS_DIR))
    outputs = comparison.run_full_comparison()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(outputs["metrics"].round(4).to_string())
    print()
    print("Statistical tests:")
    print(outputs["statistical_tests"].to_string(index=False))

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Models evaluated: {len(all_results)}")
    print(f"Plots: {RESULTS_DIR}/plots/")
    print(f"Tables: {RESULTS_DIR}/tables/")

if __name__ == "__main__":
    main()
