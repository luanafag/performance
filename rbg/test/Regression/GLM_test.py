"""Compare Gaussian identity-link GLM predictions with ordinary least squares."""

import numpy as np

from rbg.Regression import GLM
from rbg.Regression import Linear
from rbg.Utilities import ExponentialFamily


def gaussian_identity_family() -> ExponentialFamily:
    """Use independent Gaussian components with fixed unit dispersion."""
    def cumulant(theta: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(theta**2, axis=1)

    def mean(theta: np.ndarray) -> np.ndarray:
        # With this canonical Gaussian family, eta = theta = mu.
        return theta.copy()

    def curvature(theta: np.ndarray) -> np.ndarray:
        n_samples, n_components = theta.shape
        return np.broadcast_to(
            np.eye(n_components), (n_samples, n_components, n_components),
        ).copy()

    def inverse_mean(mu: np.ndarray) -> np.ndarray:
        return mu.copy()

    return ExponentialFamily(cumulant, mean, curvature, inverse_mean, phi=1.0)


class TestGaussianIdentityGLM:
    def setup_method(self) -> None:
        x = np.linspace(-2.0, 2.0, 12)
        self.X = np.column_stack((x, x**2))
        coefficients = np.array([[1.5, -0.75], [2.0, 0.5]])
        noise = np.column_stack((0.05 * np.sin(x), 0.07 * np.cos(x)))
        self.responses = self.X @ coefficients + np.array([1.0, -2.0]) + noise
        new_x = np.linspace(-1.5, 1.5, 5)
        self.X_new = np.column_stack((new_x, new_x**2))

    def check_predictions_and_dimensions(self, y: np.ndarray) -> None:
        """Check numerical predictions and preserve each response's dimensionality."""
        model = GLM(family=gaussian_identity_family(), fit_intercept=True)
        beta = model.fit(self.X, y)
        predictions = model.predict(self.X_new)
        n_parameters = self.X.shape[1] + 1

        if y.ndim == 1:
            linear = Linear(fit_intercept=True)
            linear.fit(self.X, y)
            expected = linear.predict(self.X_new)
            beta_shape = (n_parameters,)
            prediction_shape = (len(self.X_new),)
            n_components = 1
        else:
            # Linear accepts one response at a time: compare each GLM column
            # with an independently fitted ordinary least-squares model.
            reference_predictions = []
            for column in range(y.shape[1]):
                linear = Linear(fit_intercept=True)
                linear.fit(self.X, y[:, column])
                reference_predictions.append(linear.predict(self.X_new))
            expected = np.column_stack(reference_predictions)
            n_components = y.shape[1]
            beta_shape = (n_parameters, n_components)
            prediction_shape = (len(self.X_new), n_components)

        assert isinstance(beta, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        assert beta.shape == beta_shape
        assert predictions.shape == prediction_shape
        np.testing.assert_allclose(predictions, expected, rtol=1e-7, atol=1e-8)

        residuals = model.residuals()
        assert residuals.shape == y.shape
        np.testing.assert_allclose(residuals, y - model.predict(self.X))

        # Only verify inference dimensions: GLM uses inverse-Fisher covariance,
        # whereas Linear estimates the residual variance from the observations.
        inference_beta, covariance, inference_residuals = model.inference()
        assert inference_beta.shape == beta_shape
        assert covariance.shape == (n_parameters * n_components,) * 2
        assert inference_residuals.shape == y.shape

    def test_scalar_response_matches_linear(self) -> None:
        self.check_predictions_and_dimensions(self.responses[:, 0])

    def test_single_column_response_preserves_matrix_shape(self) -> None:
        self.check_predictions_and_dimensions(self.responses[:, :1])

    def test_multivariate_response_matches_separate_linear_models(self) -> None:
        self.check_predictions_and_dimensions(self.responses)
