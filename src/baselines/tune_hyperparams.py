import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pipeline import DataPipeline
from src.evaluation.metrics import PortfolioMetrics
from src.baselines.run_baselines import evaluate_model
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

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("tune")

CV_FOLDS = [
    {"train_end": "2014-12-31", "val_start": "2015-01-02", "val_end": "2016-12-31"},
    {"train_end": "2016-12-31", "val_start": "2017-01-02", "val_end": "2018-12-31"},
    {"train_end": "2018-12-31", "val_start": "2019-01-02", "val_end": "2019-12-31"},
]

OUTPUT_DIR = Path("configs/tuning_results")

def load_data(n_assets: int = 100):

    logger.info("Loading data pipeline (n_assets=%d)...", n_assets)
    pipeline = DataPipeline(
        train_start="2000-01-01",
        train_end="2019-12-31",
        test_start="2020-01-02",
        test_end="2024-12-31",
        n_assets=n_assets,
    )
    train_split, test_split = pipeline.build()

    all_assets = sorted(set(train_split.all_assets) | set(test_split.all_assets))

    returns = train_split.returns[all_assets]
    presence = (train_split.presence[all_assets] == 1.0)

    logger.info("Data loaded: %d days, %d assets", len(returns), len(all_assets))
    return returns, presence

def prepare_folds(returns: pd.DataFrame, presence: pd.DataFrame):

    folds = []
    for spec in CV_FOLDS:
        train_end = pd.Timestamp(spec["train_end"])
        val_start = pd.Timestamp(spec["val_start"])
        val_end = pd.Timestamp(spec["val_end"])

        train_ret = returns.loc[:train_end]
        train_pres = presence.loc[:train_end]
        val_ret = returns.loc[val_start:val_end]
        val_pres = presence.loc[val_start:val_end]

        if len(val_ret) < 50:
            logger.warning("Fold val %s-%s has only %d days, skipping",
                           val_start.date(), val_end.date(), len(val_ret))
            continue

        folds.append({
            "train_data": {"returns": train_ret, "presence_mask": train_pres},
            "val_data": {"returns": val_ret, "presence_mask": val_pres},
        })

    logger.info("Prepared %d CV folds", len(folds))
    return folds

def eval_on_folds(model_class, params, folds, is_ml=True,
                  train_years=3, rebalance_freq=5, smooth_alpha=None):

    sharpes = []

    for fold in folds:
        if is_ml:
            n_days = train_years * 252
            train_data = {
                "returns": fold["train_data"]["returns"].iloc[-n_days:],
                "presence_mask": fold["train_data"]["presence_mask"].iloc[-n_days:],
            }
        else:
            train_data = fold["train_data"]

        model = model_class(**params)
        if smooth_alpha is not None:
            model = SmoothedModelWrapper(model, alpha=smooth_alpha)
        try:
            ret_series, _ = evaluate_model(
                model, train_data, fold["val_data"],
                rebalance_freq=rebalance_freq,
            )
            sharpe = PortfolioMetrics.sharpe_ratio(ret_series)
            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = -2.0
        except Exception as e:
            logger.warning("Trial failed: %s", e)
            sharpe = -2.0

        sharpes.append(sharpe)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return float(np.mean(sharpes))

def xgboost_objective(trial, folds):

    params = {
        "lookback": trial.suggest_int("lookback", 10, 40, step=5),
        "top_k": trial.suggest_int("top_k", 10, 50, step=5),
        "max_weight": trial.suggest_float("max_weight", 0.03, 0.10, step=0.01),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    }
    rebalance_freq = trial.suggest_int("rebalance_freq", 13, 21, step=4)
    train_years = trial.suggest_int("train_years", 2, 5)
    smooth_alpha = trial.suggest_float("smooth_alpha", 0.1, 0.5, step=0.05)

    xgb_extra = {
        "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
    }

    model_params = {k: v for k, v in params.items()
                    if k in ("lookback", "top_k", "max_weight", "n_estimators",
                             "max_depth", "learning_rate")}

    class _TunedXGB(XGBoostModel):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.xgb_params["subsample"] = xgb_extra["subsample"]
            self.xgb_params["colsample_bytree"] = xgb_extra["colsample_bytree"]
            self.xgb_params["min_child_weight"] = trial.suggest_int(
                "min_child_weight", 5, 100, step=5)
            self.xgb_params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)
            self.xgb_params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)

    return eval_on_folds(_TunedXGB, model_params, folds,
                         is_ml=True, train_years=train_years,
                         rebalance_freq=rebalance_freq,
                         smooth_alpha=smooth_alpha)

def lightgbm_objective(trial, folds):

    params = {
        "lookback": trial.suggest_int("lookback", 10, 40, step=5),
        "top_k": trial.suggest_int("top_k", 10, 50, step=5),
        "max_weight": trial.suggest_float("max_weight", 0.03, 0.10, step=0.01),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    }
    rebalance_freq = trial.suggest_int("rebalance_freq", 13, 21, step=4)
    train_years = trial.suggest_int("train_years", 2, 5)
    smooth_alpha = trial.suggest_float("smooth_alpha", 0.1, 0.5, step=0.05)

    lgb_extra = {
        "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, step=4),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }

    model_params = {k: v for k, v in params.items()
                    if k in ("lookback", "top_k", "max_weight", "n_estimators",
                             "max_depth", "learning_rate")}

    class _TunedLGB(LightGBMModel):
        def __init__(self, **kw):
            super().__init__(**kw)
            for key in ("subsample", "colsample_bytree", "num_leaves",
                        "min_child_samples", "reg_alpha", "reg_lambda"):
                self.lgb_params[key] = lgb_extra[key]

    return eval_on_folds(_TunedLGB, model_params, folds,
                         is_ml=True, train_years=train_years,
                         rebalance_freq=rebalance_freq,
                         smooth_alpha=smooth_alpha)

def lstm_objective(trial, folds):

    params = {
        "seq_len": trial.suggest_int("seq_len", 10, 40, step=5),
        "top_k": trial.suggest_int("top_k", 10, 50, step=5),
        "max_weight": trial.suggest_float("max_weight", 0.03, 0.10, step=0.01),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6, step=0.05),
        "epochs": trial.suggest_int("epochs", 5, 25, step=5),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
    }
    rebalance_freq = trial.suggest_int("rebalance_freq", 13, 21, step=4)
    train_years = trial.suggest_int("train_years", 2, 4)
    smooth_alpha = trial.suggest_float("smooth_alpha", 0.1, 0.5, step=0.05)

    return eval_on_folds(LSTMModel, params, folds,
                         is_ml=True, train_years=train_years,
                         rebalance_freq=rebalance_freq,
                         smooth_alpha=smooth_alpha)

def transformer_objective(trial, folds):

    d_model = trial.suggest_categorical("d_model", [32, 64])
    nhead = trial.suggest_categorical("nhead", [2, 4])

    if d_model % nhead != 0:
        nhead = 2

    params = {
        "seq_len": trial.suggest_int("seq_len", 10, 40, step=5),
        "top_k": trial.suggest_int("top_k", 10, 50, step=5),
        "max_weight": trial.suggest_float("max_weight", 0.03, 0.10, step=0.01),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "epochs": trial.suggest_int("epochs", 5, 25, step=5),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
    }
    rebalance_freq = trial.suggest_int("rebalance_freq", 13, 21, step=4)
    train_years = trial.suggest_int("train_years", 2, 4)
    smooth_alpha = trial.suggest_float("smooth_alpha", 0.1, 0.5, step=0.05)

    return eval_on_folds(TransformerModel, params, folds,
                         is_ml=True, train_years=train_years,
                         rebalance_freq=rebalance_freq,
                         smooth_alpha=smooth_alpha)

def mean_variance_objective(trial, folds):

    params = {
        "lookback": trial.suggest_int("lookback", 63, 504, step=21),
        "max_weight": trial.suggest_float("max_weight", 0.02, 0.10, step=0.01),
        "risk_aversion": trial.suggest_float("risk_aversion", 0.5, 5.0, step=0.25),
        "shrinkage": trial.suggest_float("shrinkage", 0.2, 0.8, step=0.05),
        "min_history": trial.suggest_int("min_history", 30, 120, step=10),
    }

    return eval_on_folds(MeanVarianceModel, params, folds,
                         is_ml=False, rebalance_freq=5)

def risk_parity_objective(trial, folds):

    params = {
        "vol_lookback": trial.suggest_int("vol_lookback", 20, 252, step=10),
        "min_history": trial.suggest_int("min_history", 10, 60, step=5),
    }
    rebalance_freq = trial.suggest_int("rebalance_freq", 5, 21, step=4)

    return eval_on_folds(RiskParityModel, params, folds,
                         is_ml=False, rebalance_freq=rebalance_freq)

MODEL_SPECS = {
    "xgboost":       {"fn": xgboost_objective,       "n_trials": 30},
    "lightgbm":      {"fn": lightgbm_objective,       "n_trials": 30},
    "lstm":          {"fn": lstm_objective,            "n_trials": 25},
    "transformer":   {"fn": transformer_objective,     "n_trials": 25},
    "mean_variance": {"fn": mean_variance_objective,   "n_trials": 30},
    "risk_parity":   {"fn": risk_parity_objective,     "n_trials": 20},
}

def run_study(model_name: str, folds, n_trials: int | None = None):

    spec = MODEL_SPECS[model_name]
    n = n_trials or spec["n_trials"]
    objective = lambda trial: spec["fn"](trial, folds)

    logger.info("=" * 60)
    logger.info("Tuning %s — %d trials", model_name, n)
    logger.info("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=model_name,
    )
    study.optimize(objective, n_trials=n, show_progress_bar=True)

    best = study.best_trial
    logger.info("Best %s: Sharpe=%.4f", model_name, best.value)
    logger.info("Best params: %s", json.dumps(best.params, indent=2))

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "best_sharpe_cv": best.value,
        "best_params": best.params,
        "n_trials": n,
        "n_folds": len(folds),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }
    with open(output_dir / f"{model_name}_study.json", "w") as f:
        json.dump(result, f, indent=2)

    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            result["param_importance"] = importance
            with open(output_dir / f"{model_name}_study.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("Parameter importance: %s",
                        json.dumps(importance, indent=2, default=str))
        except Exception:
            pass

    return study, result

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_SPECS.keys()) + ["all"],
                        help="Which model to tune (default: all)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of trials")
    parser.add_argument("--n-assets", type=int, default=100)
    args = parser.parse_args()

    returns, presence = load_data(args.n_assets)
    folds = prepare_folds(returns, presence)

    models_to_tune = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models_to_tune:
        study, result = run_study(model_name, folds, args.n_trials)
        all_results[model_name] = result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        name: {
            "best_sharpe_cv": r["best_sharpe_cv"],
            "best_params": r["best_params"],
        }
        for name, r in all_results.items()
    }
    with open(OUTPUT_DIR / "best_params_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 70)
    for name, r in all_results.items():
        print(f"\n{name}: CV Sharpe = {r['best_sharpe_cv']:.4f}")
        for k, v in r["best_params"].items():
            print(f"  {k}: {v}")
    print("=" * 70)

if __name__ == "__main__":
    main()
