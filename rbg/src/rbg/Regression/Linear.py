import numpy as np

from .linearmodel import LinearModel


class Linear(LinearModel):
    """Ordinary least-squares regression with optional classical inference.

    Inference assumes uncorrelated errors with constant variance.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit ordinary least squares, replacing previous training results.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Target vector of shape (n_samples,).

        Returns:
            Coefficient vector, with the intercept first when enabled.

        Raises:
            ValueError: If inputs are invalid or requested inference lacks
                full column rank or positive residual degrees of freedom.
        """
        self._prepare_training_data(X, y)
        if self._fit_done == False:
            self._beta, self._residuals_square_sum, self._rank,self._singularvaluesX = np.linalg.lstsq(self._X,self._y, rcond=None)
            self._fit_done = True
        if self._inference:
            self._compute_inference()
        return self._beta

    def _compute_inference(self) -> None:
        """Compute ordinary least-squares inference without refitting."""
        if not self._inference_done:
            if self._rank != self._p or self._n <= self._p:
                raise ValueError("X must have full column rank and n must exceed p to perform inference.")
            self._residuals = self.residuals()
            self._sigma2_hat = np.dot(self._residuals, self._residuals)/(self._n - self._p)
            self._cov_matrix_hat = self._sigma2_hat * np.linalg.inv(self._X.T@self._X)
            self._inference_done = True
