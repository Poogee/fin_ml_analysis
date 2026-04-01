from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AdditionalFeatureExtractor:

    def __init__(
        self,
        momentum_windows: list[int] | None = None,
        vol_windows: list[int] | None = None,
        ma_windows: list[int] | None = None,
        vol_regime_window: int = 252,
        vol_regime_percentile: float = 75.0,
    ):
        self.momentum_windows = momentum_windows or [5, 20, 60, 252]
        self.vol_windows = vol_windows or [5, 20, 60]
        self.ma_windows = ma_windows or [20, 50]
        self.vol_regime_window = vol_regime_window
        self.vol_regime_percentile = vol_regime_percentile

    def compute_all(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:

        if prices is None:
            prices = (1 + returns).cumprod()

        per_asset_parts = []
        global_parts = []

        for w in self.momentum_windows:
            ts_mom = returns.rolling(window=w, min_periods=w).sum()
            ts_mom_z = self._cross_sectional_zscore(ts_mom)
            per_asset_parts.append(
                self._to_long(ts_mom_z, f"ts_momentum_{w}d")
            )

        for w in self.momentum_windows:
            cum_ret = returns.rolling(window=w, min_periods=w).sum()
            cs_mom = cum_ret.rank(axis=1, pct=True)
            per_asset_parts.append(
                self._to_long(cs_mom, f"cs_momentum_rank_{w}d")
            )

        for w in self.vol_windows:
            rvol = returns.rolling(window=w, min_periods=w).std() * np.sqrt(252)
            rvol_z = self._cross_sectional_zscore(rvol)
            per_asset_parts.append(
                self._to_long(rvol_z, f"realized_vol_{w}d")
            )

        vol_20d = returns.rolling(window=20, min_periods=20).std() * np.sqrt(252)
        vol_regime = vol_20d.rolling(
            window=self.vol_regime_window, min_periods=60
        ).apply(
            lambda x: (x.iloc[-1] >= np.nanpercentile(x.iloc[:-1], self.vol_regime_percentile))
            if len(x) > 1 else 0.0,
            raw=False,
        )
        per_asset_parts.append(self._to_long(vol_regime, "vol_regime_high"))

        for w in self.ma_windows:
            rolling_mean = prices.rolling(window=w, min_periods=w).mean()
            rolling_std = prices.rolling(window=w, min_periods=w).std()

            rolling_std = rolling_std.replace(0, np.nan)
            zscore = (prices - rolling_mean) / rolling_std
            zscore_clipped = zscore.clip(-3, 3)
            per_asset_parts.append(
                self._to_long(zscore_clipped, f"mean_reversion_z_{w}d")
            )

        if volumes is not None and not volumes.empty:

            vol_ma20 = volumes.rolling(window=20, min_periods=10).mean()
            vol_ma20 = vol_ma20.replace(0, np.nan)
            rel_vol = volumes / vol_ma20
            rel_vol_z = self._cross_sectional_zscore(rel_vol)
            per_asset_parts.append(self._to_long(rel_vol_z, "relative_volume"))

            vol_ma5 = volumes.rolling(window=5, min_periods=5).mean()
            vol_mom = vol_ma5 / vol_ma20
            vol_mom_z = self._cross_sectional_zscore(vol_mom)
            per_asset_parts.append(self._to_long(vol_mom_z, "volume_momentum"))

        ret_dispersion = returns.std(axis=1)
        ret_dispersion_z = (ret_dispersion - ret_dispersion.rolling(60, min_periods=20).mean()) / \
                           ret_dispersion.rolling(60, min_periods=20).std().replace(0, np.nan)
        global_parts.append(ret_dispersion_z.rename("return_dispersion_z"))
        global_parts.append(ret_dispersion.rename("return_dispersion_raw"))

        for w in self.ma_windows:
            ma = prices.rolling(window=w, min_periods=w).mean()
            breadth = (prices > ma).mean(axis=1)
            global_parts.append(breadth.rename(f"breadth_above_{w}d_ma"))

        avg_corr = self._rolling_avg_correlation(returns, window=60)
        global_parts.append(avg_corr.rename("avg_pairwise_corr"))

        mkt_ret = returns.mean(axis=1)
        for w in [5, 20, 60]:
            mkt_mom = mkt_ret.rolling(window=w, min_periods=w).sum()
            global_parts.append(mkt_mom.rename(f"market_momentum_{w}d"))

        mkt_vol = mkt_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)
        global_parts.append(mkt_vol.rename("market_vol_20d"))

        if per_asset_parts:
            per_asset_df = pd.concat(per_asset_parts, axis=1)

            per_asset_df = per_asset_df.fillna(0.0)
        else:
            per_asset_df = pd.DataFrame()

        if global_parts:
            global_df = pd.concat(global_parts, axis=1)
            global_df = global_df.fillna(0.0)
        else:
            global_df = pd.DataFrame()

        return {"per_asset": per_asset_df, "global": global_df}

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:

        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1).replace(0, np.nan)
        result = df.sub(row_mean, axis=0).div(row_std, axis=0)
        return result

    @staticmethod
    def _to_long(df: pd.DataFrame, col_name: str) -> pd.DataFrame:

        long = df.stack(future_stack=True)
        long.index.names = ["date", "asset"]
        return long.rename(col_name).to_frame()

    @staticmethod
    def _rolling_avg_correlation(
        returns: pd.DataFrame, window: int = 60
    ) -> pd.Series:

        dates = returns.index
        n_dates = len(dates)
        n_assets = returns.shape[1]
        result = pd.Series(np.nan, index=dates, dtype=np.float64)

        if n_assets < 2:
            return pd.Series(0.0, index=dates, dtype=np.float64)

        step = max(1, window // 4)

        for i in range(window, n_dates, step):
            window_data = returns.iloc[i - window:i].values
            window_data = np.nan_to_num(window_data, nan=0.0)
            corr = np.corrcoef(window_data.T)
            corr = np.nan_to_num(corr, nan=0.0)

            upper = corr[np.triu_indices(n_assets, k=1)]
            result.iloc[i] = upper.mean()

        result = result.ffill()
        return result

    def per_asset_feature_names(self, has_volumes: bool = False) -> list[str]:

        names = []
        for w in self.momentum_windows:
            names.append(f"ts_momentum_{w}d")
        for w in self.momentum_windows:
            names.append(f"cs_momentum_rank_{w}d")
        for w in self.vol_windows:
            names.append(f"realized_vol_{w}d")
        names.append("vol_regime_high")
        for w in self.ma_windows:
            names.append(f"mean_reversion_z_{w}d")
        if has_volumes:
            names.extend(["relative_volume", "volume_momentum"])
        return names

    def global_feature_names(self) -> list[str]:

        names = [
            "return_dispersion_z", "return_dispersion_raw",
        ]
        for w in self.ma_windows:
            names.append(f"breadth_above_{w}d_ma")
        names.append("avg_pairwise_corr")
        for w in [5, 20, 60]:
            names.append(f"market_momentum_{w}d")
        names.append("market_vol_20d")
        return names

    @property
    def per_asset_feature_dim(self) -> int:

        return len(self.per_asset_feature_names(has_volumes=False))

    @property
    def global_feature_dim(self) -> int:

        return len(self.global_feature_names())
