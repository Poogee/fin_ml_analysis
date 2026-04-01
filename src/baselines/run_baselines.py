import sys
import logging
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pipeline import DataPipeline
from src.evaluation.metrics import PortfolioMetrics
from src.baselines import (
    EqualWeightModel,
    RiskParityModel,
    MeanVarianceModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    TransformerModel,
)
from src.baselines.smoothed_wrapper import SmoothedModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model, train_data: dict, test_data: dict,
                   rebalance_freq: int = 5,
                   transaction_cost_bps: float = 10.0,
                   slippage_bps: float = 5.0,
                   universe_schedule: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:

    n_assets = test_data["returns"].shape[1]
    test_returns = test_data["returns"]
    test_dates = test_returns.index
    columns = list(test_returns.columns)
    total_cost_bps = transaction_cost_bps + slippage_bps
    lookback_window = 300

    model.fit(train_data)

    full_returns = pd.concat([
        train_data["returns"], test_returns
    ]).sort_index()
    full_returns = full_returns[~full_returns.index.duplicated(keep="first")]

    full_presence = None
    if "presence_mask" in test_data and "presence_mask" in train_data:
        full_presence = pd.concat([
            train_data["presence_mask"], test_data["presence_mask"]
        ]).sort_index()
        full_presence = full_presence[~full_presence.index.duplicated(keep="first")]

    from src.evaluation.backtester import _get_universe_mask

    test_locs = [full_returns.index.get_loc(d) for d in test_dates]

    current_weights = np.zeros(n_assets)
    all_returns_list = []
    all_weights = []

    for i, date in enumerate(test_dates):
        if i % rebalance_freq == 0:
            date_loc = test_locs[i]

            start = max(0, date_loc + 1 - lookback_window)
            available_data = {
                "returns": full_returns.iloc[start:date_loc + 1],
            }
            if full_presence is not None:
                available_data["presence_mask"] = full_presence.iloc[start:date_loc + 1]

            new_weights = model.predict_weights(available_data)

            if "presence_mask" in test_data:
                mask = test_data["presence_mask"].iloc[i].values.astype(bool)
                new_weights[~mask] = 0.0

            if universe_schedule:
                univ_mask = _get_universe_mask(date, universe_schedule, columns)
                new_weights[~univ_mask] = 0.0

            w_sum = new_weights.sum()
            if w_sum > 0:
                new_weights /= w_sum

            turnover = np.abs(new_weights - current_weights).sum()
            tc = turnover * total_cost_bps / 10000
            current_weights = new_weights
        else:
            tc = 0.0

        day_ret = test_returns.iloc[i].values
        port_ret = np.nansum(current_weights * day_ret) - tc
        all_returns_list.append(port_ret)
        all_weights.append(current_weights.copy())

    returns_series = pd.Series(all_returns_list, index=test_dates, name="return")
    weights_df = pd.DataFrame(all_weights, index=test_dates, columns=test_returns.columns)
    return returns_series, weights_df

def build_models() -> dict:

    xgb = XGBoostModel(
        lookback=30, top_k=40, max_weight=0.07,
        n_estimators=50, max_depth=2, learning_rate=0.0145,
    )

    xgb.xgb_params["subsample"] = 0.8
    xgb.xgb_params["colsample_bytree"] = 0.8
    xgb.xgb_params["min_child_weight"] = 50
    xgb.xgb_params["reg_alpha"] = 0.544
    xgb.xgb_params["reg_lambda"] = 5.94

    lgb = LightGBMModel(
        lookback=40, top_k=50, max_weight=0.05,
        n_estimators=50, max_depth=4, learning_rate=0.0108,
    )
    lgb.lgb_params["subsample"] = 0.9
    lgb.lgb_params["colsample_bytree"] = 0.7
    lgb.lgb_params["num_leaves"] = 27
    lgb.lgb_params["min_child_samples"] = 170
    lgb.lgb_params["reg_alpha"] = 1.74
    lgb.lgb_params["reg_lambda"] = 9.82

    return {
        "Equal Weight": EqualWeightModel(),

        "Risk Parity": RiskParityModel(vol_lookback=80, min_history=30),

        "Mean-Variance": MeanVarianceModel(
            lookback=105, max_weight=0.08, risk_aversion=2.25, shrinkage=0.8,
            min_history=30,
        ),

        "XGBoost": SmoothedModelWrapper(xgb, alpha=0.15),
        "LightGBM": SmoothedModelWrapper(lgb, alpha=0.15),

        "LSTM": SmoothedModelWrapper(
            LSTMModel(
                seq_len=15, top_k=40, max_weight=0.07,
                hidden_size=64, num_layers=3, dropout=0.25,
                epochs=5, batch_size=256, learning_rate=0.00342,
            ),
            alpha=0.25,
        ),

        "Transformer": SmoothedModelWrapper(
            TransformerModel(
                seq_len=25, top_k=40, max_weight=0.07,
                d_model=32, nhead=4, num_layers=2,
                dim_feedforward=64, dropout=0.4,
                epochs=20, batch_size=1024, learning_rate=0.00438,
            ),
            alpha=0.10,
        ),
    }

def plot_equity_curves(results: dict[str, pd.Series], output_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(14, 7))
    for name, returns in results.items():
        equity = (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=1.2)

    ax.set_title("Baseline Models — Equity Curves (Test Period)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value (starting at 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curves saved to %s", output_path)

def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--n-assets", type=int, default=100)
    parser.add_argument("--train-start", default="2000-01-01")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-02")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--rebalance-freq", type=int, default=5)
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building data pipeline...")
    pipeline = DataPipeline(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        n_assets=args.n_assets,
    )
    train_split, test_split = pipeline.build()

    all_assets = test_split.all_assets
    logger.info("Total assets in test universe (superset): %d", len(all_assets))

    merged_schedule = {
        **train_split.universe_schedule,
        **test_split.universe_schedule,
    }
    logger.info(
        "Universe schedule: %d rebalance dates (train: %d, test: %d)",
        len(merged_schedule),
        len(train_split.universe_schedule),
        len(test_split.universe_schedule),
    )

    train_data = {
        "returns": train_split.returns[all_assets],
        "presence_mask": (train_split.presence[all_assets] == 1.0),
    }
    test_data = {
        "returns": test_split.returns[all_assets],
        "presence_mask": (test_split.presence[all_assets] == 1.0),
    }

    train_data_short = {
        "returns": train_data["returns"].iloc[-1008:],
        "presence_mask": train_data["presence_mask"].iloc[-1008:],
    }

    models = build_models()
    all_returns = {}
    all_weights = {}

    model_rebalance = {
        "Equal Weight": 21,
        "Risk Parity": 21,
        "Mean-Variance": 5,
        "XGBoost": 17,
        "LightGBM": 21,
        "LSTM": 21,
        "Transformer": 17,
    }

    for name, model in models.items():
        logger.info("Running: %s", name)

        is_ml = name in ("XGBoost", "LightGBM", "LSTM", "Transformer")
        td = train_data_short if is_ml else train_data

        rebal = model_rebalance.get(name, args.rebalance_freq)
        returns, weights = evaluate_model(
            model, td, test_data,
            rebalance_freq=rebal,
            universe_schedule=merged_schedule,
        )
        all_returns[name] = returns
        all_weights[name] = weights

        metrics = PortfolioMetrics.compute_all(returns)
        logger.info("  %s: Sharpe=%.3f, Cum.Ret=%.3f, MaxDD=%.3f",
                     name, metrics["Sharpe Ratio"],
                     metrics["Cumulative Return"], metrics["Max Drawdown"])
        gc.collect()

    metrics_df = PortfolioMetrics.compute_all_df(all_returns)
    for name, w_df in all_weights.items():
        if len(w_df) > 1:
            metrics_df.loc[name, "Avg Turnover"] = PortfolioMetrics.turnover(w_df).mean()

    metrics_df = metrics_df.sort_values("Sharpe Ratio", ascending=False)

    metrics_df.to_csv(output_dir / "baseline_metrics.csv")
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS (Test Period)")
    print("=" * 80)
    print(metrics_df.round(4).to_string())
    print("=" * 80)

    plot_equity_curves(all_returns, output_dir / "equity_curves.png")

    for name, returns in all_returns.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        returns.to_frame("return").to_parquet(output_dir / f"returns_{safe_name}.parquet")

    logger.info("All baseline results saved to %s", output_dir)

if __name__ == "__main__":
    main()
