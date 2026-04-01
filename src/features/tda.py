from __future__ import annotations

import numpy as np
from ripser import ripser

class TDAFeatureExtractor:

    def __init__(
        self,
        max_homology_dim: int = 1,
        n_persistence_stats: int = 5,
    ):
        self.max_dim = max_homology_dim
        self.n_stats = n_persistence_stats

    def compute_persistence(self, returns: np.ndarray) -> dict:

        returns_clean = np.nan_to_num(returns, nan=0.0)

        corr = np.corrcoef(returns_clean.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        corr = np.clip(corr, -1.0, 1.0)

        distance = np.sqrt(2.0 * (1.0 - corr))
        np.fill_diagonal(distance, 0.0)

        result = ripser(distance, maxdim=self.max_dim, distance_matrix=True)

        return {
            "diagrams": result["dgms"],
            "distance_matrix": distance,
        }

    def extract_features(self, returns: np.ndarray) -> dict[str, np.ndarray]:

        result = self.compute_persistence(returns)
        diagrams = result["diagrams"]

        features = {}

        h0 = diagrams[0]

        h0_finite = h0[np.isfinite(h0[:, 1])]
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([0.0])

        features["betti_0"] = np.array([float(len(h0_finite))])
        features["h0_mean_lifetime"] = np.array([h0_lifetimes.mean() if len(h0_lifetimes) > 0 else 0.0])
        features["h0_max_lifetime"] = np.array([h0_lifetimes.max() if len(h0_lifetimes) > 0 else 0.0])
        features["persistence_entropy_h0"] = np.array([self._persistence_entropy(h0_lifetimes)])

        features["top_h0_lifetimes"] = self._top_k_lifetimes(h0_lifetimes)

        if self.max_dim >= 1 and len(diagrams) > 1:
            h1 = diagrams[1]
            h1_finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
            h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([0.0])

            features["betti_1"] = np.array([float(len(h1_finite))])
            features["h1_mean_lifetime"] = np.array([h1_lifetimes.mean() if len(h1_lifetimes) > 0 else 0.0])
            features["h1_max_lifetime"] = np.array([h1_lifetimes.max() if len(h1_lifetimes) > 0 else 0.0])
            features["h1_total_persistence"] = np.array([h1_lifetimes.sum()])
            features["persistence_entropy_h1"] = np.array([self._persistence_entropy(h1_lifetimes)])
            features["top_h1_lifetimes"] = self._top_k_lifetimes(h1_lifetimes)
        else:
            features["betti_1"] = np.array([0.0])
            features["h1_mean_lifetime"] = np.array([0.0])
            features["h1_max_lifetime"] = np.array([0.0])
            features["h1_total_persistence"] = np.array([0.0])
            features["persistence_entropy_h1"] = np.array([0.0])
            features["top_h1_lifetimes"] = np.zeros(self.n_stats)

        return features

    def _persistence_entropy(self, lifetimes: np.ndarray) -> float:

        if len(lifetimes) == 0 or lifetimes.sum() == 0:
            return 0.0

        total = lifetimes.sum()
        probs = lifetimes / total

        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def _top_k_lifetimes(self, lifetimes: np.ndarray) -> np.ndarray:

        if len(lifetimes) == 0:
            return np.zeros(self.n_stats)

        sorted_lt = np.sort(lifetimes)[::-1]
        result = np.zeros(self.n_stats)
        n_copy = min(len(sorted_lt), self.n_stats)
        result[:n_copy] = sorted_lt[:n_copy]
        return result

    @staticmethod
    def tda_feature_names(n_stats: int = 5) -> list[str]:

        names = [
            "betti_0", "h0_mean_lifetime", "h0_max_lifetime", "persistence_entropy_h0",
        ]
        names += [f"top_h0_lifetime_{i}" for i in range(n_stats)]
        names += [
            "betti_1", "h1_mean_lifetime", "h1_max_lifetime",
            "h1_total_persistence", "persistence_entropy_h1",
        ]
        names += [f"top_h1_lifetime_{i}" for i in range(n_stats)]
        return names
