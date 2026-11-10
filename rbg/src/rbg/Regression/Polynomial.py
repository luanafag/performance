import numpy as np

from .Linear import Linear
from .Ridge import Ridge


class Polynomial():
    """Combine per-feature polynomial powers with a selected regression model.

    Expand each feature separately, without cross-feature interactions or a
    constant column. The internal estimator handles the intercept. For example,
    degrees [2, 3] produces columns x1, x1**2, x2, x2**2, x2**3.
    """

    def __init__(self, degree=None, method="linear", fit_intercept=True, *, degrees=None):
        """Configure polynomial degrees and create the internal estimator.

        Args:
            degree: Sequence of positive integer degrees, one per input feature.
            degrees: Compatibility alias for degree; supply only one spelling.
            method: Estimator choice: 'linear', 'ridge', or 'lasso'.
            fit_intercept: Whether the internal estimator fits an intercept.

        Raises:
            ValueError: If degrees is not a nonempty one-dimensional sequence
                of positive integers, or method is not supported.
        """
        if degree is not None and degrees is not None:
            raise ValueError("Provide only one of degree or degrees.")
        self._degrees = np.asarray(degree if degree is not None else degrees)
        if (
            self._degrees.ndim != 1
            or self._degrees.size == 0
            or not np.issubdtype(self._degrees.dtype, np.integer)
            or np.any(self._degrees < 1)
        ):
            raise ValueError("degrees must be a nonempty sequence of positive integers.")

        models = {
            "linear": Linear,
            "ridge": Ridge,
        }

        if method == "lasso":
            from rbg.Regression.Lasso import Lasso

            models["lasso"] = Lasso

        if method not in models:
            raise ValueError("method must be 'linear', 'ridge', or 'lasso'.")

        self._method = method
        self._model = models[method](fit_intercept=fit_intercept)

    def get_params(self) -> dict:
        """Return polynomial degrees, estimator choice and intercept setting."""
        params = {
            "degrees": self._degrees.copy(),
            "method": self._method,
            "fit_intercept": self._model.get_params()["fit_intercept"],
        }
        if self._method in ("ridge", "lasso"):
            params["alpha"] = self._model.get_params()["alpha"]
        return params

    def set_params(self, **params):
        """Update configuration and replace any previously fitted estimator."""
        unknown = set(params) - {"degrees", "method", "fit_intercept", "alpha"}
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}.")
        if params:
            current = self.get_params()
            current.update(params)
            if not isinstance(current["fit_intercept"], bool):
                raise ValueError("fit_intercept must be a boolean.")
            alpha = current.pop("alpha", None)
            if current["method"] == "linear" and "alpha" in params:
                raise ValueError("alpha is only available for ridge or lasso.")
            if current["method"] == "linear":
                alpha = None
            self.__init__(**current)
            if alpha is not None:
                self._model.set_params(alpha=alpha)
        return self

    @property
    def coef(self) -> np.ndarray:
        """Return coefficients for expanded features, with intercept first."""
        return self._model.coef

    def _transform(self, X):
        """Expand features into their individual polynomial powers.

        Args:
            X: Feature matrix of shape (n_samples, p_features).

        Returns:
            Matrix of shape (n_samples, sum(degrees)), ordered first by input
            feature and then by increasing power. No intercept column is added.

        Raises:
            ValueError: If X is not two-dimensional or its feature count does
                not match the number of configured degrees.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self._degrees):
            raise ValueError("Provide one degree per feature.")

        return np.column_stack([
            X[:, j] ** power
            for j, degree in enumerate(self._degrees)
            for power in range(1, degree + 1)
        ])

    def fit(self, X, y, **fit_options):
        """Transform training features and fit the internal estimator.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).
            **fit_options: Estimator-specific fitting options, such as alpha
                for Ridge or Lasso. Linear regression does not accept alpha.

        Returns:
            Coefficients for the expanded features, with the intercept first
            when enabled. Ridge and Lasso undo their internal scaling.
        """
        return self._model.fit(self._transform(X), y, **fit_options)

    def predict(self, X):
        """Transform new features and return the internal model's predictions.

        Args:
            X: Matrix of shape (n_new_samples, p_features), in training order.

        Returns:
            Predicted response vector of shape (n_new_samples,).

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.predict(self._transform(X))

    def residuals(self):
        """Return the internal estimator's training residuals.

        Returns:
            Observed targets minus predictions, with shape (n_samples,).

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.residuals()

    def inference(self, **options):
        """Delegate inference for the expanded coefficients to the estimator.

        Args:
            **options: Estimator-specific inference options

        Returns:
            Coefficients, estimated error variance, coefficient covariance,
            and training residuals, as returned by the internal estimator.

        Raises:
            RuntimeError: If the internal estimator has not been fitted.
        """
        return self._model.inference(**options)
