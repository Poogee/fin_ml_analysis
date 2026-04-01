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

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.metrics import PortfolioMetrics
from src.evaluation.backtester import BacktestResult, run_walk_forward
from src.baselines.smoothed_wrapper import SmoothedModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("Building data pipeline...")
    pipeline = DataPipeline()
    train_split, test_split = pipeline.build()

    assets = test_split.all_assets
    logger.info("Test period: %d days, %d assets", test_split.n_days, len(assets))

    merged_schedule = {
        **train_split.universe_schedule,
        **test_split.universe_schedule,
    }

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

    from src.baselines.xgboost_model import XGBoostModel
    from src.baselines.lightgbm_model import LightGBMModel
    from src.baselines.lstm_model import LSTMModel
    from src.baselines.transformer_model import TransformerModel

    models = {
        "XGBoost": SmoothedModelWrapper(
            XGBoostModel(lookback=20, top_k=50, max_weight=0.05,
                         n_estimators=200, max_depth=4, learning_rate=0.05),
            alpha=0.3,
        ),
        "LightGBM": SmoothedModelWrapper(
            LightGBMModel(lookback=20, top_k=50, max_weight=0.05,
                          n_estimators=200, max_depth=4, learning_rate=0.05),
            alpha=0.3,
        ),
        "LSTM": SmoothedModelWrapper(
            LSTMModel(seq_len=20, top_k=50, max_weight=0.05,
                      hidden_size=64, num_layers=2, epochs=10, batch_size=512),
            alpha=0.3,
        ),
        "Transformer": SmoothedModelWrapper(
            TransformerModel(seq_len=20, top_k=50, max_weight=0.05,
                             d_model=64, nhead=4, num_layers=2, epochs=10, batch_size=512),
            alpha=0.3,
        ),
    }

    for name, model in models.items():
        logger.info("=" * 50)
        logger.info("Evaluating: %s (smoothed, rebalance=21d)", name)
        t0 = time.time()

        try:
            result = run_walk_forward(
                model=model,
                train_split=train_aligned,
                test_split=test_aligned,
                model_name=name,
                transaction_cost_bps=10.0,
                slippage_bps=5.0,
                rebalance_frequency=21,
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

            safe = name.replace(" ", "_")
            result.returns.to_csv(RESULTS_DIR / f"{safe}_returns.csv")
            with open(RESULTS_DIR / f"{safe}_result.pkl", "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            logger.error("FAILED: %s — %s", name, e, exc_info=True)

    logger.info("Improved ML baselines complete!")

if __name__ == "__main__":
    main()
