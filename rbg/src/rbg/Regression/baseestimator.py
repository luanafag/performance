from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    """Abstract base class with shared data storage and residual calculations."""

    _parameter_attributes = {"fit_intercept": "_intercept", "inference": "_inference"}

    def __init__(self, fit_intercept: bool = True, inference: bool = False) -> None:
        """Configure intercept fitting and inference.

        Args:
            fit_intercept: Whether to prepend a column of ones to X.
            inference: Whether to request inference calculations during fitting.
        """
        self._intercept = fit_intercept
        self._inference = inference
        self._inference_done = False
        self._residuals = None
        self._fit_done = False

    def get_params(self) -> dict:
        """Return the model configuration, excluding learned coefficients."""
        return {name: getattr(self, attribute) for name, attribute in self._parameter_attributes.items()}

    def set_params(self, **params):
        """Update configuration and invalidate a previous fit.

        Raises:
            ValueError: If a parameter name or value is invalid.
        """
        unknown = set(params) - self._parameter_attributes.keys()
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}.")
        for name in ("fit_intercept", "inference"):
            if name in params and not isinstance(params[name], bool):
                raise ValueError(f"{name} must be a boolean.")
        for name, value in params.items():
            setattr(self, self._parameter_attributes[name], value)
        if params:
            self._fit_done = False
            self._inference_done = False
            self._residuals = None
        return self

    @property
    def coef(self) -> np.ndarray:
        """Return fitted coefficients, including the intercept when enabled."""
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        return self._beta.copy()

    def _prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate and store training data, resetting previous fit results.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).

        Raises:
            ValueError: If inputs have invalid dimensions, incompatible sample
                counts, empty features, or nonnumeric or nonfinite values.
        """
        X = np.array(X)
        y = np.array(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be two-dimensional and y must be one-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")
        self._fit_done = False
        self._inference_done = False
        self._residuals = None
        self._X = X
        self._y = y
        self._X_input = self._X.copy()
        self._n, self._p = np.shape(self._X)
        if self._intercept:
                    self._X = np.column_stack((np.ones(self._n), self._X))
                    self._p += 1

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using the supplied training features and targets."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict responses for new observations using the fitted model.

        Args:
            X: Feature matrix without an automatically generated intercept column.
        """
        pass

    @abstractmethod
    def inference(self):
        """Return inference results specific to the fitted model."""
        pass

    @abstractmethod
    def _predict_mean(self, X):
        """Return predicted mean responses for residual calculations."""
        pass

    def residuals(self) -> np.ndarray:
        """Return observed targets minus predicted mean responses.

        Returns:
            Residual vector of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        if self._residuals is None:
            self._residuals = self._y - self._predict_mean(self._X_input)
        return self._residuals
