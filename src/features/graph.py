from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse import csr_matrix
import pandas as pd

class CorrelationGraph:

    def __init__(
        self,
        method: str = "knn",
        k: int = 10,
        threshold: float = 0.5,
        use_normalized_laplacian: bool = True,
        diffusion_times: list[float] | None = None,
    ):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.use_normalized = use_normalized_laplacian
        self.diffusion_times = diffusion_times or [0.5, 1.0, 5.0, 10.0]

        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._adjacency: np.ndarray | None = None
        self._laplacian: np.ndarray | None = None
        self._n_assets: int = 0

    def fit(self, returns: np.ndarray | pd.DataFrame) -> CorrelationGraph:

        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        returns_clean = np.nan_to_num(returns, nan=0.0)

        corr = np.corrcoef(returns_clean.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        self._n_assets = corr.shape[0]

        self._adjacency = self._build_adjacency(corr)

        self._compute_laplacian()
        self._eigendecompose()

        return self

    def _build_adjacency(self, corr: np.ndarray) -> np.ndarray:

        n = corr.shape[0]
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)

        if self.method == "knn":
            adj = np.zeros((n, n))
            for i in range(n):

                k_actual = min(self.k, n - 1)
                neighbors = np.argsort(abs_corr[i])[-k_actual:]
                adj[i, neighbors] = abs_corr[i, neighbors]

            adj = np.maximum(adj, adj.T)

        elif self.method == "threshold":
            adj = np.where(abs_corr >= self.threshold, abs_corr, 0.0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return adj

    def _compute_laplacian(self) -> None:

        A = self._adjacency
        D = np.diag(A.sum(axis=1))

        if self.use_normalized:

            d_inv_sqrt = np.diag(A.sum(axis=1))

            diag_vals = d_inv_sqrt.diagonal()
            nonzero = diag_vals > 0
            d_inv_sqrt_vals = np.zeros_like(diag_vals)
            d_inv_sqrt_vals[nonzero] = 1.0 / np.sqrt(diag_vals[nonzero])
            D_inv_sqrt = np.diag(d_inv_sqrt_vals)

            L = D - A
            self._laplacian = D_inv_sqrt @ L @ D_inv_sqrt
        else:
            self._laplacian = D - A

    def _eigendecompose(self) -> None:

        eigenvalues, eigenvectors = eigh(self._laplacian)

        eigenvalues = np.maximum(eigenvalues, 0.0)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    def diffuse(self, signal: np.ndarray, t: float) -> np.ndarray:

        if self._eigenvalues is None:
            raise RuntimeError("Call fit() first")

        coeffs = np.exp(-t * self._eigenvalues)

        projected = self._eigenvectors.T @ signal
        diffused = self._eigenvectors @ (coeffs * projected)

        return diffused

    def diffusion_residuals(self, signal: np.ndarray) -> np.ndarray:

        residuals = np.zeros((len(signal), len(self.diffusion_times)))
        for i, t in enumerate(self.diffusion_times):
            diffused = self.diffuse(signal, t)
            residuals[:, i] = signal - diffused
        return residuals

    def spectral_clustering(self, n_clusters: int = 2) -> np.ndarray:

        if self._eigenvectors is None:
            raise RuntimeError("Call fit() first")

        from sklearn.cluster import KMeans

        n_use = min(n_clusters, self._eigenvectors.shape[1] - 1)
        embedding = self._eigenvectors[:, 1:1 + n_use]

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embedding)

        return labels

    def cluster_distances(self, n_clusters: int = 2) -> np.ndarray:

        if self._eigenvectors is None:
            raise RuntimeError("Call fit() first")

        from sklearn.cluster import KMeans

        n_use = min(n_clusters, self._eigenvectors.shape[1] - 1)
        embedding = self._eigenvectors[:, 1:1 + n_use]

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(embedding)

        distances = np.min(
            np.linalg.norm(
                embedding[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        )
        return distances

    @property
    def algebraic_connectivity(self) -> float:

        if self._eigenvalues is None:
            raise RuntimeError("Call fit() first")
        return float(self._eigenvalues[1]) if len(self._eigenvalues) > 1 else 0.0

    @property
    def spectral_gap(self) -> float:

        if self._eigenvalues is None:
            raise RuntimeError("Call fit() first")
        lmax = self._eigenvalues[-1]
        if lmax == 0:
            return 0.0
        return float(self._eigenvalues[1] / lmax) if len(self._eigenvalues) > 1 else 0.0

    @property
    def eigenvalues(self) -> np.ndarray:

        if self._eigenvalues is None:
            raise RuntimeError("Call fit() first")
        return self._eigenvalues

    def fiedler_vector(self) -> np.ndarray:

        if self._eigenvectors is None:
            raise RuntimeError("Call fit() first")
        return self._eigenvectors[:, 1]

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:

        if self._adjacency is None:
            raise RuntimeError("Call fit() first")

        A = self._adjacency
        n = A.shape[0]

        x = np.ones(n) / n
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                return np.ones(n) / n
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        x_new = np.abs(x_new)
        s = x_new.sum()
        if s > 0:
            x_new = x_new / s
        return x_new

    def extract_features(
        self, daily_returns: np.ndarray, n_clusters: int = 4
    ) -> dict[str, np.ndarray]:

        daily_returns = np.nan_to_num(daily_returns, nan=0.0)

        return {
            "diffusion_residuals": self.diffusion_residuals(daily_returns),
            "cluster_labels": self.spectral_clustering(n_clusters),
            "cluster_distances": self.cluster_distances(n_clusters),
            "fiedler": self.fiedler_vector(),
            "eigenvector_centrality": self.eigenvector_centrality(),
            "algebraic_connectivity": np.array([self.algebraic_connectivity]),
            "spectral_gap": np.array([self.spectral_gap]),
        }
