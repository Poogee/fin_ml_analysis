import sys
import time
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import DataPipeline
from src.evaluation.metrics import PortfolioMetrics
from src.evaluation.backtester import (
    BacktestResult, run_walk_forward, datasplit_to_dict,
)
from src.evaluation.comparison import ModelComparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def load_baselines() -> dict:

    models = {}

    from src.baselines.equal_weight import EqualWeightModel
    models["Equal Weight"] = EqualWeightModel()

    from src.baselines.risk_parity import RiskParityModel
    models["Risk Parity"] = RiskParityModel()

    from src.baselines.mean_variance import MeanVarianceModel
    models["Mean-Variance (Markowitz)"] = MeanVarianceModel()

    from src.baselines.xgboost_model import XGBoostModel
    models["XGBoost"] = XGBoostModel()

    from src.baselines.lightgbm_model import LightGBMModel
    models["LightGBM"] = LightGBMModel()

    from src.baselines.lstm_model import LSTMModel
    models["LSTM"] = LSTMModel()

    from src.baselines.transformer_model import TransformerModel
    models["Transformer"] = TransformerModel()

    return models

def load_rl_models() -> dict:

    models = {}

    try:
        from src.rl.agent import RLFullModel
        models["RL Full (PPO+Graph+TDA+NLP)"] = RLFullModel()
    except (ImportError, ModuleNotFoundError):
        logger.warning("RL Full model not available yet (Task #4 dependency)")

    try:
        from src.rl.agent import RLNoGraphModel
        models["RL No Graph Features"] = RLNoGraphModel()
    except (ImportError, ModuleNotFoundError):
        logger.warning("RL No-Graph ablation not available yet")

    try:
        from src.rl.agent import RLNoSentimentModel
        models["RL No Sentiment"] = RLNoSentimentModel()
    except (ImportError, ModuleNotFoundError):
        logger.warning("RL No-Sentiment ablation not available yet")

    return models

def load_precomputed_results() -> dict[str, BacktestResult]:

    results = {}
    for pkl_file in RESULTS_DIR.glob("*_result.pkl"):
        name = pkl_file.stem.replace("_result", "").replace("_", " ")
        with open(pkl_file, "rb") as f:
            results[name] = pickle.load(f)
        logger.info("Loaded pre-computed result: %s", name)

    for csv_file in RESULTS_DIR.glob("*_returns.csv"):
        name = csv_file.stem.replace("_returns", "").replace("_", " ")
        if name in results:
            continue
        returns = pd.read_csv(csv_file, index_col=0, parse_dates=True).squeeze()
        equity = (1 + returns).cumprod()
        results[name] = BacktestResult(
            model_name=name, returns=returns, equity_curve=equity,
        )
        logger.info("Loaded pre-computed returns: %s", name)

    return results

def run_backtest_for_model(model, name, train_split, test_split,
                          universe_schedule=None):

    logger.info("=" * 40)
    logger.info("Backtesting: %s", name)
    t0 = time.time()

    result = run_walk_forward(
        model=model,
        train_split=train_split,
        test_split=test_split,
        model_name=name,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        rebalance_frequency=5,
        universe_schedule=universe_schedule,
    )

    elapsed = time.time() - t0
    metrics = PortfolioMetrics.compute_all(result.returns)
    logger.info(
        "%s done in %.1fs — Sharpe: %.3f, CumRet: %.2%%,  MaxDD: %.2%%",
        name, elapsed,
        metrics["Sharpe Ratio"],
        metrics["Cumulative Return"] * 100,
        metrics["Max Drawdown"] * 100,
    )

    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    result.returns.to_csv(RESULTS_DIR / f"{safe_name}_returns.csv")
    with open(RESULTS_DIR / f"{safe_name}_result.pkl", "wb") as f:
        pickle.dump(result, f)

    return result

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION — FULL EVALUATION")
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

    merged_schedule = {
        **train_split.universe_schedule,
        **test_split.universe_schedule,
    }
    logger.info(
        "Universe schedule: %d rebalance dates (prevents look-ahead bias)",
        len(merged_schedule),
    )

    all_models = {}
    all_models.update(load_baselines())
    all_models.update(load_rl_models())
    logger.info("Models loaded: %s", list(all_models.keys()))

    precomputed = load_precomputed_results()
    logger.info("Pre-computed results: %s", list(precomputed.keys()))

    all_results = dict(precomputed)
    for name, model in all_models.items():
        if name in all_results:
            logger.info("Skipping %s (pre-computed result exists)", name)
            continue
        try:
            result = run_backtest_for_model(
                model, name, train_split, test_split,
                universe_schedule=merged_schedule,
            )
            all_results[name] = result
        except Exception as e:
            logger.error("FAILED: %s — %s", name, e, exc_info=True)

    if not all_results:
        logger.error("No results to compare!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GENERATING COMPARISON OUTPUTS")
    print("=" * 60)

    comparison = ModelComparison(all_results, output_dir=str(RESULTS_DIR))
    outputs = comparison.run_full_comparison()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Models evaluated: {len(all_results)}")
    print(f"Plots saved to: {RESULTS_DIR}/plots/")
    print(f"Tables saved to: {RESULTS_DIR}/tables/")

if __name__ == "__main__":
    main()
