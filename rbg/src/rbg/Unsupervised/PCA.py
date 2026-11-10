import numpy as np


class PCA:
    """Principal Component Analysis via SVD.

    Projects samples onto the top principal directions of the centered data.
    """

    def __init__(self, n_components: int | float | None = None) -> None:
        """Configure the number of components to keep.

        Args:
            n_components: None for all components, an int for a fixed count,
                or a float in (0, 1) for a target cumulative variance ratio.
        """
        self.n_components = n_components
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._explained_variance: np.ndarray | None = None
        self._explained_variance_ratio: np.ndarray | None = None

    def set_params(self, n_components: int | float | None = None) -> "PCA":
        """Set the number of components to keep.

        Args:
            n_components: None for all components, an int for a fixed count,
                or a float in (0, 1) for a target cumulative variance ratio.

        Returns:
            The instance.
        """
        self.n_components = n_components
        return self

    def get_params(self) -> dict:
        """Return the model hyperparameters.

        Returns:
            Dict mapping parameter name to its value.
        """
        return {"n_components": self.n_components}
    
    @property
    def mean(self) -> np.ndarray:
        """Empirical mean of each feature, shape (n_features,)."""
        if self._mean is None:
            raise RuntimeError("Model not fitted.")
        return self._mean

    @property
    def components(self) -> np.ndarray:
        """Principal axes in feature space, shape (n_components, n_features)."""
        if self._components is None:
            raise RuntimeError("Model not fitted.")
        return self._components

    @property
    def explained_variance(self) -> np.ndarray:
        """Variance explained by each selected component, shape (n_components,)."""
        if self._explained_variance is None:
            raise RuntimeError("Model not fitted.")
        return self._explained_variance

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each selected component, shape (n_components,)."""
        if self._explained_variance_ratio is None:
            raise RuntimeError("Model not fitted.")
        return self._explained_variance_ratio

    def _pre_process(self, X: np.ndarray) -> np.ndarray:
        """Center ``X`` by the stored mean after validating and converting to float64.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, n_features), dtype float64, centered.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X is empty.")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains non-finite values.")
        if self._mean is None:
            self._mean = np.mean(X_arr, axis=0)
        elif X_arr.shape[1] != self._mean.shape[0]:
            raise ValueError(f"The data do not match: expected {self._mean.shape[0]} features, got {X_arr.shape[1]}.")
        return X_arr - self._mean

    def _svd_flip(self, V: np.ndarray) -> np.ndarray:
        """Implement the sign criterion for SVD decompositions.

        Args:
            V: ndarray of shape (n_singular, n_features).

        Returns:
            ndarray of the same shape as ``V``.
        """
        max_abs_cols = np.argmax(np.abs(V), axis=1)
        signs = np.sign(V[np.arange(len(V)), max_abs_cols])
        return V * signs.reshape(-1, 1)

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit the model on ``X``.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            The fitted instance.
        """
        # reset so a second fit doesn't reuse the previous mean
        self._mean = None
        X_centered = self._pre_process(X)
        n, p = X_centered.shape

        _, S, V = np.linalg.svd(X_centered, full_matrices=False)
        V = self._svd_flip(V)

        if n > 1:
            explained_variance = S**2 / (n - 1)
        else:
            explained_variance = np.zeros(p, dtype=np.float64)

        total = explained_variance.sum()
        if total > 0:
            explained_variance_ratio = explained_variance / total
        else:
            explained_variance_ratio = np.zeros_like(explained_variance)

        nc = self.n_components
        if isinstance(nc, int):
            if nc <= 0:
                raise ValueError("n_components must be positive.")
            if nc > min(n, p):
                raise ValueError("n_components cannot exceed min(n_samples, n_features).")
            k = nc
        elif isinstance(nc, float):
            if not 0 < nc < 1:
                raise ValueError("n_components as float must be in (0, 1).")
            cum_var = np.cumsum(explained_variance_ratio)
            k = int(np.argmax(cum_var >= nc)) + 1
        else:
            k = p

        self._components = V[:k, :]
        self._explained_variance = explained_variance[:k]
        self._explained_variance_ratio = explained_variance_ratio[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` onto the principal axes.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, n_components).
        """
        if self._components is None:
            raise RuntimeError("Model not fitted.")
        X_centered = self._pre_process(X)
        return X_centered @ self._components.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return its projection onto the principal axes.

        Args:
            X: ndarray of shape (n_samples, n_features).

        Returns:
            ndarray of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map ``X`` from component space back to the original feature space.

        Args:
            X: ndarray of shape (n_samples, n_components).

        Returns:
            ndarray of shape (n_samples, n_features).
        """
        if self._components is None:
            raise RuntimeError("Model not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if X_arr.shape[1] != self._components.shape[0]:
            raise ValueError(f"expected {self._components.shape[0]} components, got {X_arr.shape[1]}.")
        return X_arr @ self._components + self._mean
