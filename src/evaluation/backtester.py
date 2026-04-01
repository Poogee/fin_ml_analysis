from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.pipeline import DataSplit

@runtime_checkable
class PortfolioModel(Protocol):

    def fit(self, train_data: dict) -> None:

        ...

    def predict_weights(self, current_data: dict) -> np.ndarray:

        ...

@dataclass
class BacktestConfig:

    train_window: int = 504
    test_window: int = 63
    step_size: int = 63
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rebalance_frequency: int = 5
    risk_free_rate: float = 0.0
    expanding: bool = False

@dataclass
class BacktestResult:

    model_name: str
    returns: pd.Series = field(default_factory=pd.Series)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    window_metrics: list = field(default_factory=list)

class RollingBacktester:

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def _compute_transaction_costs(self, old_weights: np.ndarray,
                                    new_weights: np.ndarray) -> float:

        turnover = np.abs(new_weights - old_weights).sum()
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        return turnover * total_cost_bps / 10000

    def run(self, model: PortfolioModel, data: dict,
            model_name: str = "Model") -> BacktestResult:

        returns_df = data["returns"]
        dates = returns_df.index
        n_dates = len(dates)
        n_assets = returns_df.shape[1]

        cfg = self.config
        all_returns = []
        all_weights = []
        all_dates = []

        start = 0
        while start + cfg.train_window + cfg.test_window <= n_dates:
            train_start = 0 if cfg.expanding else start
            train_end = start + cfg.train_window
            test_end = min(train_end + cfg.test_window, n_dates)

            train_data = self._slice_data(data, train_start, train_end)
            model.fit(train_data)

            current_weights = np.zeros(n_assets)
            for t in range(train_end, test_end):

                if (t - train_end) % cfg.rebalance_frequency == 0:

                    available_data = self._slice_data(data, train_start, t + 1)
                    new_weights = model.predict_weights(available_data)

                    if "presence_mask" in data:
                        prow = data["presence_mask"].iloc[t].values
                        mask = prow.astype(bool) if prow.dtype == bool else (prow == 1.0)
                        new_weights[~mask] = 0.0
                        w_sum = new_weights.sum()
                        if w_sum > 0:
                            new_weights = new_weights / w_sum

                    tc = self._compute_transaction_costs(current_weights, new_weights)
                    current_weights = new_weights
                else:
                    tc = 0.0

                day_returns = returns_df.iloc[t].values
                port_return = np.nansum(current_weights * day_returns) - tc
                all_returns.append(port_return)
                all_weights.append(current_weights.copy())
                all_dates.append(dates[t])

            start += cfg.step_size

        result_df = pd.DataFrame({
            "date": all_dates,
            "return": all_returns,
        })
        weights_df = pd.DataFrame(all_weights, index=all_dates,
                                   columns=returns_df.columns)

        result_df = result_df.drop_duplicates(subset="date", keep="first")
        result_df = result_df.set_index("date").sort_index()
        weights_df = weights_df[~weights_df.index.duplicated(keep="first")].sort_index()

        returns_series = result_df["return"]
        equity = (1 + returns_series).cumprod()

        return BacktestResult(
            model_name=model_name,
            returns=returns_series,
            weights_history=weights_df,
            equity_curve=equity,
        )

    def _slice_data(self, data: dict, start: int, end: int) -> dict:

        sliced = {}
        for key, val in data.items():
            if isinstance(val, (pd.DataFrame, pd.Series)):
                sliced[key] = val.iloc[start:end]
            else:
                sliced[key] = val
        return sliced

    def run_multiple(self, models: dict[str, PortfolioModel],
                     data: dict) -> dict[str, BacktestResult]:

        results = {}
        for name, model in models.items():
            print(f"Running backtest: {name}...")
            results[name] = self.run(model, data, model_name=name)
            print(f"  -> {len(results[name].returns)} days, "
                  f"cumulative return: {results[name].equity_curve.iloc[-1]:.4f}")
        return results

def datasplit_to_dict(split: DataSplit) -> dict:

    return {
        "returns": split.returns,
        "prices": split.prices,
        "log_returns": split.log_returns,
        "market_caps": split.market_caps,
        "presence_mask": split.presence,
    }

def _get_universe_mask(
    date: pd.Timestamp,
    universe_schedule: dict[pd.Timestamp, list[str]],
    all_columns: list,
) -> np.ndarray:

    valid_dates = [d for d in universe_schedule if d <= date]
    if not valid_dates:

        valid_dates = [min(universe_schedule)]
    latest = max(valid_dates)
    current_assets = set(universe_schedule[latest])
    return np.array([col in current_assets for col in all_columns])

def run_walk_forward(
    model: PortfolioModel,
    train_split: DataSplit,
    test_split: DataSplit,
    model_name: str = "Model",
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    rebalance_frequency: int = 5,
    max_lookback: int = 504,
    universe_schedule: dict[pd.Timestamp, list[str]] | None = None,
) -> BacktestResult:

    train_data = datasplit_to_dict(train_split)
    model.fit(train_data)

    test_returns = test_split.returns
    test_presence = test_split.presence
    dates = test_returns.index
    n_assets = test_returns.shape[1]
    columns = list(test_returns.columns)

    if universe_schedule is None:
        if test_split.universe_schedule:
            universe_schedule = {
                **train_split.universe_schedule,
                **test_split.universe_schedule,
            }

    tc_bps = transaction_cost_bps + slippage_bps
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

            w_sum = new_weights.sum()
            if w_sum > 0:
                new_weights = new_weights / w_sum

            turnover = np.abs(new_weights - current_weights).sum()
            tc = turnover * tc_bps / 10000
            current_weights = new_weights
        else:
            tc = 0.0

        port_ret = np.nansum(current_weights * test_returns_np[i]) - tc
        all_returns.append(port_ret)

    returns_series = pd.Series(all_returns, index=dates, name="return")
    equity = (1 + returns_series).cumprod()

    return BacktestResult(
        model_name=model_name,
        returns=returns_series,
        equity_curve=equity,
    )
