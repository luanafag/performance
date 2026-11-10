import numpy as np

from .baseestimator import BaseEstimator
from rbg.Utilities import FisherScoring
from .linearmodel import LinearModel

class GLM(BaseEstimator):

    """Scalar or multivariate generalized linear model with a canonical link.

    This model is intended as a foundation for logistic regression. The current
    implementation uses theta = X @ beta and a fixed family dispersion.
    Multivariate coefficients are ordered by response component when flattened:
    beta.ravel(order="F"). Fisher information must use the same order.
    """

    _parameter_attributes = {
        **BaseEstimator._parameter_attributes,
        "family": "_family",
        "tol": "_tol",
        "max_iter": "_max_iter",
    }

    def __init__(
        self, family, fit_intercept=True, inference=False, tol=1e-6, max_iter=100
    ):
        """Configure the exponential family and solver settings.

        Args:
            family: ExponentialFamily instance accepting natural parameters of
                shape (n_samples, n_components), including one component for
                scalar responses.
            fit_intercept: Whether to add an intercept column to the design.
            inference: Whether to compute coefficient covariance during fitting.
            tol: Convergence tolerance passed to FisherScoring.
            max_iter: Maximum number of solver iterations.
        """
        super().__init__(fit_intercept=fit_intercept, inference=inference)
        self._family = family
        self._tol = tol
        self._max_iter = max_iter

    def set_params(self, **params):
        if "tol" in params:
            tol = params["tol"]
            if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol) or tol <= 0:
                raise ValueError("tol must be finite and positive.")
        if "max_iter" in params:
            iterations = params["max_iter"]
            if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations <= 0:
                raise ValueError("max_iter must be a positive integer.")
        return super().set_params(**params)

    def _prepare_training_data(self, X, y):
        """Prepare scalar or vector responses without changing other estimators."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim not in (1, 2):
            raise ValueError("X must be a matrix and y must be a vector or matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data must not be empty.")
        self._X_input = X.copy()
        self._X = self._X_input.copy()
        self._y = y.copy()
        self._n, self._p = self._X.shape
        self._n_components = 1 if y.ndim == 1 else y.shape[1]
        if self._intercept:
            self._X = np.column_stack((np.ones(self._n), self._X))
            self._p += 1
        self._fit_done = False
        self._inference_done = False
        self._residuals = None

    def fit(self, X, y):
        """Fit coefficients and obtain final Fisher information from the solver.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Response vector (n_samples,) or matrix (n_samples, n_components),
                in the family's domain.

        Returns:
            Coefficients of shape (p,) for vector y or (p, n_components) for
            matrix y. Here p includes the intercept when enabled.

        Raises:
            ValueError: If training inputs fail the shared data validation.

        Notes:
            Requires an implemented FisherScoring.solve returning coefficients
            and total information evaluated at the final coefficients. For
            multivariate responses, the solver must assemble information blocks
            in component-major order, matching beta.ravel(order="F").
        """
        self._prepare_training_data(X, y)
        beta0 = np.zeros(self._p) if self._y.ndim == 1 else np.zeros((self._p, self._n_components))
        beta, fisher = FisherScoring.solve(self._X,self._y,self._family,beta0,self._tol,self._max_iter)
        beta, fisher = np.asarray(beta), np.asarray(fisher)
        self._beta, self._fisher = beta, fisher
        self._fit_done = True
        if self._inference:
            self._compute_inference()
        return self._beta

    def inference(self):
        """Return fitted coefficients, approximate covariance, and residuals.

        Returns:
            A tuple of coefficients, inverse-Fisher covariance, and training
            response residuals. Covariance is an asymptotic approximation
            under standard maximum-likelihood regularity conditions.
            Multivariate covariance uses component-major coefficient order.

        Raises:
            RuntimeError: If the model has not been fitted.
            numpy.linalg.LinAlgError: If Fisher information is singular.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        if not self._inference_done:
            self._inference = True
            self._compute_inference()
        return self._beta, self._cov_matrix_hat, self._residuals

    def predict(self, X):
        """Return predicted conditional means on the response scale.

        Args:
            X: Matrix of shape (n_new_samples, p_features), in training order
                and without an automatically generated intercept column.

        Returns:
            Mean response vector (n_new_samples,) or matrix
            (n_new_samples, n_components), matching the training response
            dimensionality. Apply the family's mean function to X @ beta.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has an incompatible shape.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._X_input.shape[1]:
            raise ValueError("X must be a matrix with the same feature count as the training data.")
        if self._intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        theta = X @ self._beta
        if theta.ndim == 1:
            return self._family.mean(theta[:, None])[:, 0]
        return self._family.mean(theta)

    def _predict_mean(self, X):
        """Return GLM means even if a classifier overrides predict with labels."""
        return GLM.predict(self, X)

    def _compute_inference(self):
        """Compute inverse-Fisher covariance and training response residuals.

        Fisher information must include dispersion scaling and be evaluated at
        the final coefficients. No additional dispersion factor is applied.

        Raises:
            RuntimeError: If the model has not been fitted.
            numpy.linalg.LinAlgError: If Fisher information is singular.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")

        if not self._inference_done:
            self._cov_matrix_hat = np.linalg.solve(
                self._fisher,
                np.eye(self._beta.size),
            )
            self._residuals = self.residuals()
            self._inference_done = True
