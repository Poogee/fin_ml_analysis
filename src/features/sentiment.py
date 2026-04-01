from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PER_TICKER_CACHE = Path(__file__).resolve().parents[2] / "data" / "sentiment_per_ticker.parquet"

@dataclass
class SentimentFeatures:

    global_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    per_asset_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    global_names: list[str] = field(default_factory=list)
    per_asset_names: list[str] = field(default_factory=list)

def build_sentiment_features(
    daily_sentiment: pd.DataFrame,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    short_window: int = 5,
    long_window: int = 20,
    beta_window: int = 60,
    per_ticker_sentiment: pd.DataFrame | None = None,
    permno_list: list[int] | None = None,
) -> SentimentFeatures:

    T, N = returns.shape

    sent = daily_sentiment.reindex(dates).ffill()
    sent["sentiment_mean"] = sent.get("sentiment_mean", pd.Series(0.0, index=dates)).fillna(0.0)
    sent["sentiment_std"] = sent.get("sentiment_std", pd.Series(0.0, index=dates)).fillna(0.0)
    sent["sentiment_positive_frac"] = sent.get("sentiment_positive_frac", pd.Series(0.33, index=dates)).fillna(0.33)
    sent["sentiment_negative_frac"] = sent.get("sentiment_negative_frac", pd.Series(0.33, index=dates)).fillna(0.33)
    sent["news_count"] = sent.get("news_count", pd.Series(0.0, index=dates)).fillna(0.0)

    global_feats = _build_global_features(sent, short_window, long_window)
    global_names = [
        "sent_mean", "sent_std",
        f"sent_ma{short_window}", f"sent_ma{long_window}",
        f"sent_momentum_{short_window}d", f"sent_momentum_{long_window}d",
        "sent_breadth", "news_volume",
        f"news_volume_change_{short_window}d", "sent_dispersion_change",
    ]

    if per_ticker_sentiment is not None and permno_list is not None:
        pa_feats, pa_names = _build_per_asset_ticker_features(
            per_ticker_sentiment, dates, permno_list, T, N,
            short_window, long_window,
        )
        logger.info("Using per-ticker sentiment: %d features per asset", pa_feats.shape[2])
    else:
        pa_feats, pa_names = _build_per_asset_beta_features(
            sent, returns, beta_window, long_window,
        )
        logger.info("Using market-level sentiment proxy: %d features per asset", pa_feats.shape[2])

    return SentimentFeatures(
        global_features=global_feats,
        per_asset_features=pa_feats,
        global_names=global_names,
        per_asset_names=pa_names,
    )

def _build_global_features(
    sent: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> np.ndarray:

    T = len(sent)
    n_global = 10
    feats = np.zeros((T, n_global), dtype=np.float32)

    sm = sent["sentiment_mean"].values.astype(np.float32)
    ss = sent["sentiment_std"].values.astype(np.float32)
    sp = sent["sentiment_positive_frac"].values.astype(np.float32)
    sn = sent["sentiment_negative_frac"].values.astype(np.float32)
    nv = sent["news_count"].values.astype(np.float32)

    for t in range(T):
        feats[t, 0] = sm[t]
        feats[t, 1] = ss[t]
        if t >= short_window:
            feats[t, 2] = sm[t - short_window:t].mean()
        if t >= long_window:
            feats[t, 3] = sm[t - long_window:t].mean()
        if t >= 2 * short_window:
            feats[t, 4] = sm[t - short_window:t].mean() - sm[t - 2 * short_window:t - short_window].mean()
        if t >= 2 * long_window:
            feats[t, 5] = sm[t - long_window:t].mean() - sm[t - 2 * long_window:t - long_window].mean()
        feats[t, 6] = sp[t] - sn[t]
        feats[t, 7] = nv[t]
        if t >= short_window:
            feats[t, 8] = nv[t] - nv[max(0, t - short_window):t].mean()
            feats[t, 9] = ss[t] - ss[max(0, t - short_window):t].mean()

    return feats

def _build_per_asset_ticker_features(
    per_ticker_sent: pd.DataFrame,
    dates: pd.DatetimeIndex,
    permno_list: list[int],
    T: int,
    N: int,
    short_window: int = 5,
    long_window: int = 20,
) -> tuple[np.ndarray, list[str]]:

    n_pa = 6
    feats = np.zeros((T, N, n_pa), dtype=np.float32)

    if isinstance(per_ticker_sent.index, pd.MultiIndex):
        sent_wide = per_ticker_sent["ticker_sentiment_mean"].unstack(level="permno")
        count_wide = per_ticker_sent["ticker_news_count"].unstack(level="permno")
    else:
        sent_wide = per_ticker_sent
        count_wide = None

    sent_wide = sent_wide.reindex(dates).ffill(limit=5).fillna(0.0)
    if count_wide is not None:
        count_wide = count_wide.reindex(dates).fillna(0.0)

    permno_to_col = {int(c): i for i, c in enumerate(permno_list)}

    for j, perm in enumerate(permno_list):
        if perm not in sent_wide.columns:
            continue

        s = sent_wide[perm].values.astype(np.float32)
        nc = count_wide[perm].values.astype(np.float32) if count_wide is not None else np.zeros(T, dtype=np.float32)

        for t in range(T):

            feats[t, j, 0] = s[t]

            feats[t, j, 4] = np.log1p(nc[t])

            if t >= short_window:
                feats[t, j, 1] = s[t - short_window:t + 1].mean()

            if t >= long_window:
                feats[t, j, 2] = s[t - long_window:t + 1].mean()

            if t >= 2 * short_window:
                ma_now = s[t - short_window:t + 1].mean()
                ma_prev = s[t - 2 * short_window:t - short_window + 1].mean()
                feats[t, j, 3] = ma_now - ma_prev

            if t >= long_window:
                feats[t, j, 5] = s[t] - s[t - long_window:t].mean()

    names = [
        "ticker_sentiment", "ticker_sent_ma5", "ticker_sent_ma20",
        "ticker_sent_momentum_5d", "ticker_news_volume", "ticker_sent_surprise",
    ]
    return feats, names

def _build_per_asset_beta_features(
    sent: pd.DataFrame,
    returns: np.ndarray,
    beta_window: int,
    long_window: int,
) -> tuple[np.ndarray, list[str]]:

    T, N = returns.shape
    n_pa = 2
    feats = np.zeros((T, N, n_pa), dtype=np.float32)

    sm = sent["sentiment_mean"].values.astype(np.float64)

    for t in range(beta_window, T):
        window_sent = sm[t - beta_window:t]
        sent_mean = window_sent.mean()
        sent_var = np.var(window_sent)

        if sent_var < 1e-10:
            continue

        sent_demeaned = window_sent - sent_mean

        for j in range(N):
            window_ret = returns[t - beta_window:t, j].astype(np.float64)
            ret_mean = np.nanmean(window_ret)
            cov = np.nanmean((window_ret - ret_mean) * sent_demeaned)
            beta = cov / sent_var
            predicted = beta * (sm[t] - sent_mean)
            residual = returns[t, j] - predicted

            feats[t, j, 0] = np.float32(beta)
            feats[t, j, 1] = np.float32(residual)

    names = ["sentiment_beta", "sentiment_adj_return"]
    return feats, names

GLOBAL_SENTIMENT_DIM = 10
PER_ASSET_SENTIMENT_DIM = 6
PER_ASSET_SENTIMENT_DIM_FALLBACK = 2
