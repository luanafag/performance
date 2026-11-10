import numpy as np

from .linearmodel import LinearModel


class Ridge(LinearModel):
    """Ridge regression with feature scaling and original-scale coefficients.

    Inference treats alpha as fixed. The residual variance estimate may be
    biased by shrinkage and does not establish ordinary t tests or intervals.
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
        """Estimate coefficients and calculate inference statistics if requested.

        Each call replaces the previous fit and cached inference results.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Target vector of shape (n_samples,).
            alpha: Finite, nonnegative L2 penalty on scaled feature coefficients.

        Returns:
            Coefficient vector in the original feature scale, with the intercept
            first when enabled.

        Raises:
            ValueError: If alpha or training inputs are invalid, or requested
                inference fails its rank or residual degrees-of-freedom checks.
        """
        self._alpha = self._alpha if alpha is None else alpha
        if not isinstance(self._alpha, (int,float,np.integer,np.floating)):
            raise ValueError("Alpha must be a numeric scalar.")
        if not np.isfinite(self._alpha) or self._alpha < 0:
            raise ValueError("Alpha must be finite and nonnegative.")
        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._D = np.eye(self._p)
            if self._intercept == True:
                self._D[0,0] = 0
            self._means = self._X_input.mean(axis=0) if self._intercept else np.zeros(self._X_input.shape[1])
            self._stds = self._X_input.std(axis=0)
            self._stds = np.where(self._stds == 0, 1.0, self._stds)
            self._Z = (self._X_input - self._means) / self._stds
            if self._intercept:
                self._Z = np.column_stack((np.ones(self._n), self._Z))
            X_aug = np.vstack((self._Z, np.sqrt(self._alpha) * self._D))
            y_aug = np.concatenate((self._y, np.zeros(self._p)))
            self._beta, self._residuals_square_sum, _,_ = np.linalg.lstsq(X_aug,y_aug, rcond=None)
            # Convert coefficients back to the original feature scale.
            self._T = np.diag(1.0 / self._stds)
            if self._intercept:
                self._T = np.eye(self._p)
                self._T[0, 1:] = -self._means / self._stds
                self._T[1:, 1:] = np.diag(1.0 / self._stds)
            self._beta = self._T @ self._beta
            self._fit_done = True
        if self._inference:
            self._compute_inference()
        return self._beta

    def _compute_inference(self) -> None:
        """Estimate error variance and original-scale coefficient covariance.

        Use the standardized design and effective residual degrees of freedom.
        The current implementation requires full column rank in that design.

        Raises:
            ValueError: If the standardized design is rank deficient or the
                residual degrees of freedom are not positive.
        """
        if not self._inference_done:
            if np.linalg.matrix_rank(self._Z) < self._p:
                 raise ValueError("Z must have full column rank.")
            A = self._Z.T @ self._Z + self._alpha*self._D
            M, _,_,_ = np.linalg.lstsq(A,self._Z.T, rcond=None)
            H = self._Z@M
            df = self._n - 2*np.trace(H) + np.trace(H.T@H)
            if not df > 0:
                raise ValueError("Residual degrees of freedom must be positive.")

            e = self.residuals()

            self._sigma2_hat = np.dot(e,e)/df

            self._cov_matrix_hat_padronized = self._sigma2_hat * (M @ M.T)

            self._cov_matrix_hat = self._T @ self._cov_matrix_hat_padronized @ self._T.T
            self._inference_done = True
