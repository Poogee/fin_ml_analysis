from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.graph import CorrelationGraph
from src.features.tda import TDAFeatureExtractor
from src.features.additional import AdditionalFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:

    per_asset: pd.DataFrame = field(default_factory=pd.DataFrame)
    global_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates: list = field(default_factory=list)
    asset_names: list = field(default_factory=list)

    def get_observation(
        self, date: pd.Timestamp, assets: list[str] | None = None
    ) -> dict[str, np.ndarray]:

        if assets is None:
            assets = self.asset_names

        if isinstance(self.per_asset.index, pd.MultiIndex):
            try:
                pa = self.per_asset.loc[date]
                pa = pa.reindex(assets).fillna(0.0).values
            except KeyError:
                n_features = self.per_asset.shape[1] if len(self.per_asset) > 0 else 1
                pa = np.zeros((len(assets), n_features))
        else:
            pa = np.zeros((len(assets), 1))

        if date in self.global_features.index:
            gf = self.global_features.loc[date].values
        else:
            n_global = self.global_features.shape[1] if len(self.global_features) > 0 else 1
            gf = np.zeros(n_global)

        return {
            "per_asset_features": pa.astype(np.float32),
            "global_features": gf.astype(np.float32),
        }

class FeaturePipeline:

    def __init__(
        self,
        corr_window: int = 60,
        graph_method: str = "knn",
        graph_k: int = 10,
        graph_threshold: float = 0.5,
        diffusion_times: list[float] | None = None,
        n_clusters: int = 4,
        tda_max_dim: int = 1,
        n_persistence_stats: int = 5,
        recompute_freq: int = 5,
        use_additional_features: bool = True,
        additional_kwargs: dict | None = None,
    ):
        self.corr_window = corr_window
        self.n_clusters = n_clusters
        self.recompute_freq = recompute_freq
        self.use_additional_features = use_additional_features

        self.graph = CorrelationGraph(
            method=graph_method,
            k=graph_k,
            threshold=graph_threshold,
            diffusion_times=diffusion_times or [0.5, 1.0, 5.0, 10.0],
        )
        self.tda = TDAFeatureExtractor(
            max_homology_dim=tda_max_dim,
            n_persistence_stats=n_persistence_stats,
        )

        self._diffusion_times = diffusion_times or [0.5, 1.0, 5.0, 10.0]

        if use_additional_features:
            self.additional = AdditionalFeatureExtractor(**(additional_kwargs or {}))
        else:
            self.additional = None

    def compute_all(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
        log_progress: bool = True,
    ) -> FeatureSet:

        dates = returns.index
        assets = list(returns.columns)
        n_assets = len(assets)
        n_dates = len(dates)

        pa_cols = (
            [f"diffusion_residual_t{t}" for t in self._diffusion_times]
            + ["cluster_label", "cluster_distance", "fiedler_value"]
        )
        n_pa_features = len(pa_cols)

        global_cols = self._global_feature_names()

        per_asset_records = []
        global_records = []
        feature_dates = []

        prev_lambda2 = None
        prev_betti1 = None

        cached_graph_features = None
        cached_tda_features = None
        last_recompute = -self.recompute_freq

        for i in range(self.corr_window, n_dates):
            date = dates[i]

            need_recompute = (i - last_recompute) >= self.recompute_freq

            if need_recompute:

                window = returns.iloc[i - self.corr_window:i].values

                try:

                    self.graph.fit(window)
                    today_returns = returns.iloc[i].values
                    today_returns = np.nan_to_num(today_returns, nan=0.0)
                    cached_graph_features = self.graph.extract_features(
                        today_returns, n_clusters=self.n_clusters
                    )

                    cached_tda_features = self.tda.extract_features(window)
                    last_recompute = i

                except Exception as e:
                    logger.warning("Feature computation failed at %s: %s", date, e)
                    continue
            else:

                if cached_graph_features is not None:
                    today_returns = returns.iloc[i].values
                    today_returns = np.nan_to_num(today_returns, nan=0.0)

                    cached_graph_features["diffusion_residuals"] = \
                        self.graph.diffusion_residuals(today_returns)

            if cached_graph_features is None or cached_tda_features is None:
                continue

            gf = cached_graph_features
            tf = cached_tda_features

            pa_data = np.zeros((n_assets, n_pa_features))
            pa_data[:, :len(self._diffusion_times)] = gf["diffusion_residuals"]
            pa_data[:, len(self._diffusion_times)] = gf["cluster_labels"]
            pa_data[:, len(self._diffusion_times) + 1] = gf["cluster_distances"]
            pa_data[:, len(self._diffusion_times) + 2] = gf["fiedler"]

            for j, asset in enumerate(assets):
                per_asset_records.append((date, asset, *pa_data[j]))

            lambda2 = float(gf["algebraic_connectivity"][0])
            sgap = float(gf["spectral_gap"][0])

            delta_lambda2 = lambda2 - prev_lambda2 if prev_lambda2 is not None else 0.0
            betti1_val = float(tf["betti_1"][0])
            delta_betti1 = betti1_val - prev_betti1 if prev_betti1 is not None else 0.0

            prev_lambda2 = lambda2
            prev_betti1 = betti1_val

            global_row = [
                lambda2, sgap,
                float(tf["betti_0"][0]), betti1_val,
                float(tf["h0_mean_lifetime"][0]), float(tf["h0_max_lifetime"][0]),
                float(tf["h1_mean_lifetime"][0]), float(tf["h1_max_lifetime"][0]),
                float(tf["h1_total_persistence"][0]),
                float(tf["persistence_entropy_h0"][0]),
                float(tf["persistence_entropy_h1"][0]),
                delta_lambda2, delta_betti1,
            ]
            global_records.append((date, *global_row))
            feature_dates.append(date)

            if log_progress and (i - self.corr_window) % 100 == 0:
                logger.info(
                    "Features computed: %d/%d dates (%.1f%%)",
                    i - self.corr_window, n_dates - self.corr_window,
                    100 * (i - self.corr_window) / max(1, n_dates - self.corr_window),
                )

        if per_asset_records:
            pa_df = pd.DataFrame(
                per_asset_records,
                columns=["date", "asset"] + pa_cols,
            )
            pa_df = pa_df.set_index(["date", "asset"])
        else:
            pa_df = pd.DataFrame()

        if global_records:
            gf_df = pd.DataFrame(
                global_records,
                columns=["date"] + global_cols,
            ).set_index("date")
        else:
            gf_df = pd.DataFrame()

        if self.additional is not None:
            logger.info("Computing additional features (momentum, vol, breadth)...")
            add_feats = self.additional.compute_all(
                returns, prices=prices, volumes=volumes
            )
            add_pa = add_feats["per_asset"]
            add_gf = add_feats["global"]

            if not pa_df.empty and not add_pa.empty:
                pa_df = pa_df.join(add_pa, how="left").fillna(0.0)
            elif not add_pa.empty:
                pa_df = add_pa.fillna(0.0)

            if not gf_df.empty and not add_gf.empty:
                gf_df = gf_df.join(add_gf, how="left").fillna(0.0)
            elif not add_gf.empty:
                gf_df = add_gf.fillna(0.0)

            logger.info("Additional features merged: +%d per-asset, +%d global",
                        add_pa.shape[1] if not add_pa.empty else 0,
                        add_gf.shape[1] if not add_gf.empty else 0)

        logger.info("Feature pipeline complete: %d dates, %d assets", len(feature_dates), n_assets)

        return FeatureSet(
            per_asset=pa_df,
            global_features=gf_df,
            dates=feature_dates,
            asset_names=assets,
        )

    def _global_feature_names(self) -> list[str]:

        return [
            "algebraic_connectivity", "spectral_gap",
            "betti_0", "betti_1",
            "h0_mean_lifetime", "h0_max_lifetime",
            "h1_mean_lifetime", "h1_max_lifetime",
            "h1_total_persistence",
            "persistence_entropy_h0", "persistence_entropy_h1",
            "delta_lambda2", "delta_betti1",
        ]

    @property
    def per_asset_feature_dim(self) -> int:

        base = len(self._diffusion_times) + 3
        if self.additional is not None:
            base += self.additional.per_asset_feature_dim
        return base

    @property
    def global_feature_dim(self) -> int:

        base = len(self._global_feature_names())
        if self.additional is not None:
            base += self.additional.global_feature_dim
        return base
