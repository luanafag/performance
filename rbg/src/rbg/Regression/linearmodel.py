from abc import abstractmethod

import numpy as np

from .baseestimator import BaseEstimator


class LinearModel(BaseEstimator):
    """Abstract base for linear predictors with model-specific fitting and inference.

    Subclasses store coefficients in the original feature scale, with the
    intercept first when enabled. Prediction and inference access are shared.
    """

    @abstractmethod
    def _compute_inference(self) -> None:
        """Compute the inference statistics specific to the subclass."""
        pass

    def inference(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Calculate and return the subclass's inference results.

        Returns:
            A tuple containing the coefficients, estimated error variance,
            estimated coefficient covariance matrix, and residual vector.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If the subclass's inference requirements are not met.
            NotImplementedError: If the subclass does not implement inference.
        """
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")
        if not self._inference_done:
            self._inference = True
            self._compute_inference()
        return self._beta, self._sigma2_hat, self._cov_matrix_hat, self._residuals

    def _predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Return linear predictions as the estimated mean response."""
        return self.predict(X)

    def predict(self,X:np.ndarray) -> np.ndarray:
        """Predict responses using the fitted coefficients.

        Args:
            X: Matrix of shape (n_new_samples, p_features), with features in
                training order and without an automatically generated intercept
                column. The number of samples may differ from training.

        Returns:
            Predicted response vector of shape (n_new_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X is not two-dimensional, contains nonnumeric or
                nonfinite values, or has a different feature count from training.
        """
        if self._fit_done == False:
                    raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X)
        if 2 != X.ndim:
            raise ValueError("X must be a two-dimensional matrix.")
        if X.shape[1] != self._X_input.shape[1]:
            raise ValueError("X must contain the same number of features as the training data.")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        if self._intercept == True:
            X = np.column_stack((np.ones(np.shape(X)[0]), X))
        return  X @ self._beta
