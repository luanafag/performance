import numpy as np

from rbg.Utilities import Resample
from .linearmodel import LinearModel
from rbg.Utilities import CoordinateDescent


class Lasso(LinearModel, CoordinateDescent):
    """Lasso regression using the CoordinateDescent solver supplied externally.

    Assumed solver contract: CoordinateDescent.solve(X, y, alpha, fit_intercept)
    is a static or class method that returns a
    converged coefficient vector in the supplied feature scale, including the
    intercept first when enabled. It minimizes 0.5 * squared_error + alpha *
    sum(abs(feature_coefficients)), leaving the intercept unpenalized.
    CoordinateDescent must be imported before this class is defined; any
    required solver settings must be initialized according to its interface.
    """

    _parameter_attributes = {**LinearModel._parameter_attributes, "alpha": "_alpha"}

    def __init__(self, fit_intercept=True, inference=False, alpha=1.0):
        super().__init__(fit_intercept=fit_intercept, inference=inference)
        self.set_params(alpha=alpha)

    def set_params(self, **params):
        if "alpha" in params:
            alpha = params["alpha"]
            if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.integer, np.floating)) or not np.isfinite(alpha) or alpha < 0:
                raise ValueError("Alpha must be finite and nonnegative.")
        return super().set_params(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Prepare scaled features and delegate the L1 fit to CoordinateDescent.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Target vector of shape (n_samples,).
            alpha: Finite, nonnegative penalty strength.

        Returns:
            Coefficients in the original feature scale,
            with the intercept first when enabled.

        Raises:
            ValueError: If inputs or alpha are invalid, or the solver returns
                coefficients with an invalid shape or nonfinite values.
            NotImplementedError: If inference is requested.
        """
        self._alpha = self._alpha if alpha is None else alpha
        if not isinstance(self._alpha, (int,float,np.integer,np.floating)):
            raise ValueError("Alpha must be a numeric scalar.")
        if not np.isfinite(self._alpha) or self._alpha < 0:
            raise ValueError("Alpha must be finite and nonnegative.")

        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._means = self._X_input.mean(axis=0) if self._intercept else np.zeros(self._X_input.shape[1])
            self._stds = self._X_input.std(axis=0)
            self._stds = np.where(self._stds == 0, 1.0, self._stds)
            self._Z = (self._X_input - self._means) / self._stds
            if self._intercept:
                self._Z = np.column_stack((np.ones(self._n), self._Z))
            beta_scaled = CoordinateDescent.solve(self._Z,self._y,self._alpha,self._intercept)
            self._T = np.diag(1.0 / self._stds)
            if self._intercept:
                self._T = np.eye(self._p)
                self._T[0, 1:] = -self._means / self._stds
                self._T[1:, 1:] = np.diag(1.0 / self._stds)
            self._beta = self._T @ beta_scaled
            self._fit_done = True
            e = self.residuals()
            self._residuals_square_sum = np.dot(e,e)
        if self._inference:
            self._compute_inference()
        return self._beta


    def inference(
        self, n_bootstrap: int = 1000, seed: int | None = None
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Configure bootstrap inference and use the inherited inference interface.

        Args:
            n_bootstrap: Number of bootstrap repetitions, at least two.
            seed: Nonnegative random seed, or None for unseeded sampling.

        Returns:
            Once bootstrap is complete, coefficients, estimated error variance,
            coefficient covariance, and training residuals.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the number of repetitions or seed is invalid.
        """
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
            raise ValueError("n_bootstrap must be an integer greater than or equal to two.")
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None.")
        if n_bootstrap != getattr(self, "_n_bootstrap", 1000) or seed != getattr(self, "_seed", None):
            self._inference_done = False
        self._n_bootstrap = n_bootstrap
        self._seed = seed
        return super().inference()

    def _compute_inference(self) -> None:
        """Estimate coefficient covariance using paired samples from Resample.

        Resample with replacement, applying each index array to both X and y.
        Use fresh models with inference disabled to preserve the original fit.
        Estimate error variance from the original residuals and active parameter
        count. This approximation may be biased and does not guarantee valid
        confidence intervals for Lasso coefficients.
        """
        if self._inference_done:
            return
        self._seed = getattr(self, "_seed", None)
        self._n_bootstrap = getattr(self, "_n_bootstrap", 1000)
        features_beta = self._beta[1:] if self._intercept else self._beta
        self._d = np.count_nonzero(features_beta) + int(self._intercept)
        if self._n <= self._d:
            raise ValueError("Residual degrees of freedom must be positive.")
        betas_boot = np.zeros((self._n_bootstrap, self._p))
        samples = Resample(
            X=self._X_input,
            size=self._n,
            n_resamples=int(self._n_bootstrap),
            random_state=self._seed,
            replace=True,
        )
        for i, (train_indices, _) in enumerate(samples):
            X_boot = self._X_input[train_indices]
            y_boot = self._y[train_indices]
            model_boot = Lasso(fit_intercept=self._intercept, inference=False)
            betas_boot[i] = model_boot.fit(X_boot, y_boot, self._alpha)
        self._residuals = e = self.residuals()
        self._sigma2_hat = np.dot(e,e)/(self._n-self._d)
        self._cov_matrix_hat = np.atleast_2d(np.cov(betas_boot, rowvar=False, ddof=1))
        self._inference_done = True
