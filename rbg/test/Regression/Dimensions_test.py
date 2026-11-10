"""Dimensional contracts for public and internal regression functions.

GLM fitting itself is covered in GLM_test.py. Here a known fitted state
isolates prediction/inference dimensions from the unfinished Fisher solver.
Lasso is imported inside its test so its missing solver does not block others.
"""

import numpy as np
import pytest

from rbg.Regression import GLM
from rbg.Regression import Linear
from rbg.Regression import Ridge
from rbg.Utilities import ExponentialFamily


def gaussian_family() -> ExponentialFamily:
    return ExponentialFamily(
        b=lambda theta: np.sum(theta**2, axis=1) / 2,
        b_prime=lambda theta: theta.copy(),
        b_second=lambda theta: np.broadcast_to(
            np.eye(theta.shape[1]), (len(theta), theta.shape[1], theta.shape[1]),
        ).copy(),
        h_inverse=lambda mu: mu.copy(),
    )


class TestRegressionDimensions:

    @pytest.fixture(autouse=True)
    def _subtest_context(self, subtests):
        self.subtests = subtests

    def setup_method(self) -> None:
        rng = np.random.default_rng(321)
        self.X = rng.normal(size=(24, 2))
        self.y = 2 + self.X @ np.array([1.5, -0.5]) + rng.normal(scale=0.1, size=24)

    def check_linear_model_dimensions(self, estimator: type) -> None:
        for intercept in (True, False):
            for n_features in (1, 2):
                with self.subtests.test(estimator=estimator.__name__, intercept=intercept, features=n_features):
                    X = self.X[:, :n_features]
                    p = n_features + int(intercept)
                    model = estimator(fit_intercept=intercept)
                    assert model._prepare_training_data(X, self.y) is None
                    assert model._X.shape == (len(X), p)
                    assert model._X_input.shape == X.shape
                    assert model._y.shape == self.y.shape
                    assert model.fit(X, self.y).shape == (p,)
                    for n in (1, 5):
                        assert model.predict(X[:n]).shape == (n,)
                        assert model._predict_mean(X[:n]).shape == (n,)
                    assert model.residuals().shape == (len(X),)
                    options = {"n_bootstrap": 4, "seed": 42} if estimator.__name__ == "Lasso" else {}
                    beta, variance, covariance, residuals = model.inference(**options)
                    assert beta.shape == (p,)
                    assert np.ndim(variance) == 0
                    assert covariance.shape == (p, p)
                    assert residuals.shape == (len(X),)
                    assert model._compute_inference() is None
                    assert model._cov_matrix_hat.shape == (p, p)
                    for invalid in (np.ones(n_features), np.ones((2, n_features + 1)), np.ones((2, n_features, 1))):
                        with pytest.raises(ValueError):
                            model.predict(invalid)

    def test_linear_function_dimensions(self) -> None:
        self.check_linear_model_dimensions(Linear)

    def test_ridge_function_dimensions(self) -> None:
        self.check_linear_model_dimensions(Ridge)

    def test_lasso_function_dimensions(self) -> None:
        from rbg.Regression.Lasso import Lasso

        self.check_linear_model_dimensions(Lasso)

    def test_training_rejects_incompatible_dimensions(self) -> None:
        for estimator in (Linear, Ridge):
            for X, y in ((self.X[:, 0], self.y), (self.X, self.y[:, None]),
                         (self.X, self.y[:-1]), (np.ones((24, 2, 1)), self.y)):
                with self.subtests.test(estimator=estimator.__name__, X=X.shape, y=y.shape):
                    with pytest.raises(ValueError):
                        estimator()._prepare_training_data(X, y)

    def test_glm_preparation_prediction_residual_and_inference_dimensions(self) -> None:
        responses = (self.y, self.y[:, None], np.column_stack((self.y, -self.y)))
        for intercept in (True, False):
            for y in responses:
                with self.subtests.test(intercept=intercept, response_shape=y.shape):
                    model = GLM(gaussian_family(), fit_intercept=intercept)
                    assert model._prepare_training_data(self.X, y) is None
                    p = self.X.shape[1] + int(intercept)
                    k = 1 if y.ndim == 1 else y.shape[1]
                    assert model._X.shape == (len(self.X), p)
                    assert model._X_input.shape == self.X.shape
                    assert model._y.shape == y.shape
                    assert model._n_components == k
                    # Known fitted state, without substituting an implementation
                    # for the solver or claiming that fit has been verified.
                    beta_shape = (p,) if y.ndim == 1 else (p, k)
                    model._beta = np.ones(beta_shape)
                    model._fisher = np.eye(p * k)
                    model._fit_done = True
                    for n in (1, 5):
                        expected = (n,) if y.ndim == 1 else (n, k)
                        assert model.predict(self.X[:n]).shape == expected
                        assert model._predict_mean(self.X[:n]).shape == expected
                    assert model.residuals().shape == y.shape
                    assert model._compute_inference() is None
                    beta, covariance, residuals = model.inference()
                    assert beta.shape == beta_shape
                    assert covariance.shape == (p * k, p * k)
                    assert residuals.shape == y.shape
                    for invalid in (np.ones(2), np.ones((3, 3)), np.ones((3, 2, 1))):
                        with pytest.raises(ValueError):
                            model.predict(invalid)

    def test_glm_preparation_rejects_incompatible_dimensions(self) -> None:
        for X, y in ((self.X[:, 0], self.y), (self.X, np.ones((24, 1, 1))),
                     (self.X, self.y[:-1]), (self.X, np.empty((24, 0)))):
            with self.subtests.test(X=X.shape, y=y.shape):
                with pytest.raises(ValueError):
                    GLM(gaussian_family())._prepare_training_data(X, y)
