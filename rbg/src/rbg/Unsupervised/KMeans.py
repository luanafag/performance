import numpy as np


class KMeans:
    """Partitions samples into k clusters by minimizing within-cluster squared distances.

    Initial centroids come either from K-Means++ or from explicit coordinates passed to the constructor.
    """

    def __init__(
        self,
        k: int = 10,
        centroid_selection: str | np.ndarray | list = "Kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        """Initialize the KMeans model.

        Args:
            k: Number of clusters to form.
            centroid_selection: Initialization method ('Kmeans++') or a list/array of initial centroid coordinates.
            max_iter: Maximum number of iterations for the algorithm.
            tol: Stop when the largest per-centroid shift falls below this value.
            seed: Seed for the random number generator.
        """
        self.k = int(k)
        self.centroid_selection = centroid_selection
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = seed

        self._centroids: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._wss: float | None = None

    def set_params(self, **params) -> "KMeans":
        """Set hyperparameters (only the ones passed are changed).

        Args:
            **params: Any of ``k``, ``centroid_selection``, ``max_iter``, ``tol``, ``seed``.

        Returns:
            The instance.

        Raises:
            ValueError: If a parameter name is unknown.
        """
        for name, value in params.items():
            if name not in ("k", "centroid_selection", "max_iter", "tol", "seed"):
                raise ValueError(f"unknown parameter '{name}'.")
            setattr(self, name, value)
        return self

    def get_params(self) -> dict:
        """Return the hyperparameters.

        Returns:
            Dict mapping parameter name to its value.
        """
        return {
            "k": self.k,
            "centroid_selection": self.centroid_selection,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "seed": self.seed,
        }
    
    @property
    def centroids(self) -> np.ndarray:
        """Centroid coordinates learned during fit.

        Returns:
            ndarray of shape (k, n_features).
        """
        if self._centroids is None:
            raise RuntimeError("Model not fitted.")
        return self._centroids

    @property
    def labels(self) -> np.ndarray:
        """Cluster label of each training sample, shape (n_samples,)."""
        if self._labels is None:
            raise RuntimeError("Model not fitted.")
        return self._labels

    @property
    def clusters(self) -> np.ndarray:
        """Alias for :attr:`labels`."""
        return self.labels

    @property
    def wss(self) -> float:
        """Within-cluster sum of squares, the K-Means objective.

        Returns:
            float, non-negative.
        """
        if self._wss is None:
            raise RuntimeError("Model not fitted.")
        return self._wss

    def _pre_process(self, X: np.ndarray) -> np.ndarray:
        """Convert ``X`` to a float64 matrix and reject shapes/dtypes the fit cannot handle.

        Args:
            X: Array-like with one row per sample.

        Returns:
            ndarray of shape (n_samples, n_features), dtype float64.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X is empty.")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains non-finite values.")
        return X_arr

    def _calc_distances_sq(self, X: np.ndarray) -> np.ndarray:
        """Squared Euclidean distances from each point to each centroid.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, k), non-negative.
        """
        dists = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2.0 * (X @ self._centroids.T)
            + np.sum(self._centroids**2, axis=1)
        )
        # forces non-negative output, the empty-cluster handler depends on it
        return np.maximum(dists, 0.0)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Select centroids from explicit coordinates or K-Means++.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (k, n_features).
        """
        n, p = X.shape

        if isinstance(self.centroid_selection, str):
            if self.centroid_selection not in ("Kmeans++", "Kmean++"):
                raise ValueError("centroid_selection must be 'Kmeans++' or coordinates.")
            use_kmeanspp = True
        else:
            use_kmeanspp = False
        
        if not use_kmeanspp:
            centroids = np.asarray(self.centroid_selection, dtype=np.float64)
            if centroids.shape != (self.k, p):
                raise ValueError(f"expected centroids of shape ({self.k}, {p}).")
            if not np.isfinite(centroids).all():
                raise ValueError("Centroids contain non-finite values.")
            return centroids.copy()

        rng = np.random.default_rng(self.seed)
        centroids = np.empty((self.k, p))

        centroids[0] = X[int(rng.integers(n))]
        min_distances_sq = np.full(n, np.inf)

        for k_idx in range(1, self.k):
            diff = X - centroids[k_idx - 1]
            dist_sq = np.sum(diff * diff, axis=1)
            min_distances_sq = np.minimum(min_distances_sq, dist_sq)

            total = float(min_distances_sq.sum())
            if total <= 0.0:
                raise ValueError("Not enough distinct points for k clusters.")

            probs = min_distances_sq / total
            probs = probs / probs.sum()
            centroids[k_idx] = X[int(rng.choice(n, p=probs))]

        return centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means to ``X`` and store centroids, labels and WSS.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            The fitted instance.
        """
        X = self._pre_process(X)
        n, p = X.shape

        if self.k > n:
            raise ValueError("k cannot exceed the number of samples.")

        self._centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            distances_sq = self._calc_distances_sq(X)
            new_labels = np.argmin(distances_sq, axis=1)
            min_dists_sq = distances_sq[np.arange(n), new_labels]

            new_centroids = np.empty_like(self._centroids)
            min_dists_sq_work = min_dists_sq.copy()
            for i in range(self.k):
                members = X[new_labels == i]
                if members.shape[0] > 0:
                    new_centroids[i] = members.mean(axis=0)
                else:
                    far = int(np.argmax(min_dists_sq_work))
                    new_centroids[i] = X[far]
                    # prevents two empty clusters from stealing the same point
                    min_dists_sq_work[far] = -np.inf

            shift = float(np.max(np.linalg.norm(new_centroids - self._centroids, axis=1)))
            self._centroids = new_centroids

            if shift < self.tol:
                break

        final = self._calc_distances_sq(X)
        self._labels = np.argmin(final, axis=1)
        self._wss = float(np.sum(final[np.arange(n), self._labels]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of ``X`` to its nearest centroid.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples,) with cluster indices in [0, k).
        """
        if self._centroids is None:
            raise RuntimeError("Model not fitted.")
        X = self._pre_process(X)
        if X.shape[1] != self._centroids.shape[1]:
            raise ValueError(f"expected {self._centroids.shape[1]} features, got {X.shape[1]}.")
        return np.argmin(self._calc_distances_sq(X), axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distance from each row of ``X`` to each centroid.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, k), non-negative.
        """
        if self._centroids is None:
            raise RuntimeError("Model not fitted.")
        X = self._pre_process(X)
        if X.shape[1] != self._centroids.shape[1]:
            raise ValueError(f"expected {self._centroids.shape[1]} features, got {X.shape[1]}.")
        return np.sqrt(self._calc_distances_sq(X))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return the cluster index of each row.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples,) with cluster indices in [0, k).
        """
        return self.fit(X).labels

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return the Euclidean distance from each row to each centroid.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, k), non-negative.
        """
        self.fit(X)
        return np.sqrt(self._calc_distances_sq(np.asarray(X, dtype=np.float64)))
