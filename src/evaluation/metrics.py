import numpy as np
import pandas as pd

class PortfolioMetrics:

    TRADING_DAYS_PER_YEAR = 252

    @staticmethod
    def cumulative_return(returns: pd.Series) -> float:

        return (1 + returns).prod() - 1

    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:

        total = (1 + returns).prod()
        n_years = len(returns) / periods_per_year
        if n_years <= 0:
            return 0.0
        return total ** (1 / n_years) - 1

    @staticmethod
    def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:

        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                     periods_per_year: int = 252) -> float:

        excess = returns - risk_free_rate / periods_per_year
        vol = excess.std()
        if vol < 1e-12:
            return 0.0
        return (excess.mean() / vol) * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return -drawdowns.min()

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:

        ann_ret = PortfolioMetrics.annualized_return(returns, periods_per_year)
        mdd = PortfolioMetrics.max_drawdown(returns)
        if mdd == 0:
            return 0.0
        return ann_ret / mdd

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                      periods_per_year: int = 252) -> float:

        excess = returns - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return (excess.mean() / downside.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 252,
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252) -> pd.Series:

        excess = returns - risk_free_rate / periods_per_year
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()

        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
        return rolling_sharpe.dropna()

    @staticmethod
    def turnover(weights: pd.DataFrame) -> pd.Series:

        return weights.diff().abs().sum(axis=1).iloc[1:]

    @staticmethod
    def bootstrap_sharpe_ci(
        returns: pd.Series,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        block_size: int = 21,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        seed: int = 42,
    ) -> dict:

        rng = np.random.RandomState(seed)
        n = len(returns)
        excess = (returns - risk_free_rate / periods_per_year).values

        n_blocks = (n + block_size - 1) // block_size
        sharpe_samples = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            starts = rng.randint(0, n, size=n_blocks)
            indices = np.concatenate([
                np.arange(s, s + block_size) % n for s in starts
            ])[:n]
            sample = excess[indices]
            std = sample.std()
            if std < 1e-12:
                sharpe_samples[b] = 0.0
            else:
                sharpe_samples[b] = (sample.mean() / std) * np.sqrt(periods_per_year)

        alpha = 1 - confidence
        point_est = PortfolioMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        return {
            "sharpe": point_est,
            "ci_lower": float(np.percentile(sharpe_samples, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(sharpe_samples, 100 * (1 - alpha / 2))),
            "confidence": confidence,
            "std_error": float(sharpe_samples.std()),
        }

    @staticmethod
    def compute_all(returns: pd.Series, risk_free_rate: float = 0.0,
                    periods_per_year: int = 252) -> dict:

        pm = PortfolioMetrics
        return {
            "Cumulative Return": pm.cumulative_return(returns),
            "Annualized Return": pm.annualized_return(returns, periods_per_year),
            "Annualized Volatility": pm.annualized_volatility(returns, periods_per_year),
            "Sharpe Ratio": pm.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            "Calmar Ratio": pm.calmar_ratio(returns, periods_per_year),
            "Sortino Ratio": pm.sortino_ratio(returns, risk_free_rate, periods_per_year),
            "Max Drawdown": pm.max_drawdown(returns),
        }

    @staticmethod
    def compute_all_df(results: dict[str, pd.Series],
                       risk_free_rate: float = 0.0) -> pd.DataFrame:

        rows = {}
        for name, rets in results.items():
            rows[name] = PortfolioMetrics.compute_all(rets, risk_free_rate)
        return pd.DataFrame(rows).T
