from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.backtester import run_walk_forward
from src.evaluation.metrics import PortfolioMetrics
from src.baselines.mean_variance import MeanVarianceModel
from src.rl.train_tilt_v2 import TiltPPO, filter_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_rl"

class MVOWrapper:

    def __init__(self, lookback=105, max_weight=0.08, risk_aversion=2.25,
                 shrinkage=0.8, min_history=30):
        self.mvo = MeanVarianceModel(
            lookback=lookback, max_weight=max_weight,
            risk_aversion=risk_aversion, shrinkage=shrinkage,
            min_history=min_history,
        )

    def fit(self, train_data):
        pass

    def predict_weights(self, current_data):
        return self.mvo.predict_weights(current_data)

class WeightBlendModel:

    def __init__(self, model_a, model_b, alpha=0.5, max_weight=0.15):
        self.model_a = model_a
        self.model_b = model_b
        self.alpha = alpha
        self.max_weight = max_weight

    def fit(self, train_data):
        self.model_a.fit(train_data)
        self.model_b.fit(train_data)

    def predict_weights(self, current_data):
        w_a = self.model_a.predict_weights(current_data)
        w_b = self.model_b.predict_weights(current_data)

        w = self.alpha * w_a + (1 - self.alpha) * w_b

        w = np.maximum(w, 0)
        w = np.minimum(w, self.max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            n = len(w)
            w = np.ones(n) / n
        return w

class AdaptiveBlendModel:

    def __init__(self, models: dict, lookback=60, max_weight=0.15):
        self.models = models
        self.lookback = lookback
        self.max_weight = max_weight
        self._returns_history = {name: [] for name in models}
        self._weights_history = {name: [] for name in models}

    def fit(self, train_data):
        for name, model in self.models.items():
            model.fit(train_data)

    def predict_weights(self, current_data):
        returns = current_data["returns"]
        n_assets = returns.shape[1]

        model_weights = {}
        for name, model in self.models.items():
            try:
                model_weights[name] = model.predict_weights(current_data)
            except Exception:
                model_weights[name] = np.ones(n_assets) / n_assets

        if not any(len(v) > 20 for v in self._returns_history.values()):
            blend_w = np.zeros(n_assets)
            for w in model_weights.values():
                blend_w += w / len(model_weights)
        else:

            scores = {}
            for name in self.models:
                recent = self._returns_history[name][-self.lookback:]
                if len(recent) >= 20:
                    arr = np.array(recent)
                    std = arr.std()
                    scores[name] = arr.mean() / (std + 1e-8) if std > 0 else 0
                else:
                    scores[name] = 0

            max_score = max(scores.values())
            exp_scores = {name: np.exp(10 * (s - max_score))
                          for name, s in scores.items()}
            total = sum(exp_scores.values())

            blend_w = np.zeros(n_assets)
            for name in self.models:
                blend_w += (exp_scores[name] / total) * model_weights[name]

        if len(returns) >= 2:
            last_ret = returns.iloc[-1].values
            for name in self.models:
                if name in self._weights_history and self._weights_history[name]:
                    prev_w = self._weights_history[name][-1]
                    model_ret = np.nansum(prev_w * last_ret)
                    self._returns_history[name].append(model_ret)

        for name, w in model_weights.items():
            self._weights_history[name].append(w.copy())

        blend_w = np.maximum(blend_w, 0)
        blend_w = np.minimum(blend_w, self.max_weight)
        w_sum = blend_w.sum()
        if w_sum > 0:
            blend_w /= w_sum
        else:
            blend_w = np.ones(n_assets) / n_assets
        return blend_w

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
    return result, metrics

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (n_assets=100)...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()
    train_split, test_split = filter_splits(train_split, test_split)
    n = train_split.n_assets
    logger.info("Train: %d days, %d assets", train_split.n_days, n)

    all_results = {}
    TARGET = 0.89

    logger.info("Testing pure MVO baseline...")
    mvo_base = MVOWrapper(lookback=105, max_weight=0.08, risk_aversion=2.25, shrinkage=0.8)
    _, m = train_eval(mvo_base, "MVO_base", train_split, test_split)
    all_results["MVO_base"] = m

    mvo_wide = MVOWrapper(lookback=105, max_weight=0.15, risk_aversion=2.25, shrinkage=0.8)
    _, m = train_eval(mvo_wide, "MVO_wide", train_split, test_split)
    all_results["MVO_wide"] = m

    mvo_agg = MVOWrapper(lookback=105, max_weight=0.12, risk_aversion=1.5, shrinkage=0.6)
    _, m = train_eval(mvo_agg, "MVO_agg_wide", train_split, test_split)
    all_results["MVO_agg_wide"] = m

    for alpha in [0.7, 0.8, 0.9]:
        name = f"Blend_MVO{int(alpha*100)}_Tilt{int((1-alpha)*100)}"
        model = WeightBlendModel(
            model_a=MVOWrapper(lookback=105, max_weight=0.08, risk_aversion=2.25, shrinkage=0.8),
            model_b=TiltPPO(tilt_scale=0.10, timesteps=500_000, lookback=10, max_weight=0.15),
            alpha=alpha,
        )
        _, m = train_eval(model, name, train_split, test_split)
        all_results[name] = m

    model = WeightBlendModel(
        model_a=MVOWrapper(lookback=105, max_weight=0.08, risk_aversion=2.25, shrinkage=0.8),
        model_b=MVOWrapper(lookback=60, max_weight=0.12, risk_aversion=1.5, shrinkage=0.6),
        alpha=0.5,
    )
    _, m = train_eval(model, "Blend_MVO_x_MVOagg", train_split, test_split)
    all_results["Blend_MVO_x_MVOagg"] = m

    model = AdaptiveBlendModel({
        "MVO": MVOWrapper(lookback=105, max_weight=0.08, risk_aversion=2.25, shrinkage=0.8),
        "TiltPPO": TiltPPO(tilt_scale=0.10, timesteps=500_000, lookback=10, max_weight=0.15),
    })
    _, m = train_eval(model, "AdaptiveBlend_MVO_Tilt", train_split, test_split)
    all_results["AdaptiveBlend_MVO_Tilt"] = m

    print("\n" + "=" * 100)
    print("ENSEMBLE BLEND RESULTS (n_assets=100)")
    print("=" * 100)
    print(f"{'Model':<35s} | {'Sharpe':>7s} | {'Return':>9s} | {'MaxDD':>8s} | {'Calmar':>7s} | {'Sortino':>7s}")
    print("-" * 100)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["Sharpe Ratio"], reverse=True)
    for name, m in sorted_results:
        flag = " ***" if m["Sharpe Ratio"] >= TARGET else ""
        print(f"  {name:<33s} | {m['Sharpe Ratio']:7.3f} | "
              f"{m['Cumulative Return']*100:8.2f}% | "
              f"{m['Max Drawdown']*100:7.2f}% | "
              f"{m['Calmar Ratio']:7.3f} | "
              f"{m['Sortino Ratio']:7.3f}{flag}")
    print("=" * 100)

    best = sorted_results[0]
    print(f"\nBest: {best[0]} — Sharpe {best[1]['Sharpe Ratio']:.3f}")
    if best[1]["Sharpe Ratio"] >= TARGET:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {TARGET - best[1]['Sharpe Ratio']:.3f}")

    pd.DataFrame(all_results).T.to_csv(RESULTS_DIR / "blend_summary.csv")

if __name__ == "__main__":
    main()
