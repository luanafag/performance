"""Check imports and working estimators after splitting the regression module."""

import numpy as np
import pytest

from rbg.Classification.Logistic import Logistic
from rbg.Regression import BaseEstimator
from rbg.Regression import GLM
from rbg.Regression import Linear
from rbg.Regression import LinearModel
from rbg.Regression import Polynomial
from rbg.Regression import Ridge


class TestRegressionModules:
    """Check the public regression estimators with pytest."""

    @pytest.fixture(autouse=True)
    def _subtest_context(self, subtests):
        self.subtests = subtests

    def test_linear_fit_prediction_and_inference(self) -> None:
        """Preserve fitting, prediction and inference with or without intercept."""
        X = np.arange(1.0, 9.0).reshape(-1, 1)
        for fit_intercept in (True, False):
            with self.subtests.test(fit_intercept=fit_intercept):
                y = 3.0 * X[:, 0] + (2.0 if fit_intercept else 0.0)
                model = Linear(fit_intercept=fit_intercept, inference=True)
                assert isinstance(model, LinearModel)
                expected = [2.0, 3.0] if fit_intercept else [3.0]
                np.testing.assert_allclose(model.fit(X, y), expected, atol=1e-12)
                np.testing.assert_allclose(model.predict(X), y, atol=1e-12)
                np.testing.assert_allclose(model.residuals(), 0, atol=1e-12)
                beta, variance, covariance, residuals = model.inference()
                np.testing.assert_allclose(beta, expected, atol=1e-12)
                assert variance < 1e-20
                assert covariance.shape == (len(expected), len(expected))
                assert residuals.shape == y.shape

    def test_ridge_regularization(self) -> None:
        """Ridge still shrinks coefficients and preserves the intercept."""
        X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
        y = 2.0 + 3.0 * X[:, 0]
        model = Ridge()
        # This implementation minimizes RSS + alpha*||scaled coefficients||².
        np.testing.assert_allclose(model.fit(X, y, alpha=4.0), [2.0, 1.5], atol=1e-12)
        np.testing.assert_allclose(model.predict([[0.0]]), [2.0], atol=1e-12)

    def test_polynomial_without_lasso_dependency(self) -> None:
        """Polynomial OLS/Ridge work without the unfinished CoordinateDescent."""
        X = np.linspace(-2.0, 2.0, 15).reshape(-1, 1)
        y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 0]**2
        for method in ("linear", "ridge"):
            with self.subtests.test(method=method):
                model = Polynomial([2], method=method)
                options = {"alpha": 0.0} if method == "ridge" else {}
                np.testing.assert_allclose(model.fit(X, y, **options), [1, 2, -0.5], atol=1e-12)
                np.testing.assert_allclose(model.predict(X), y, atol=1e-12)

    def test_logistic_uses_extracted_glm(self) -> None:
        """Classification initializes against the same GLM base definition."""
        model = Logistic(classes=[0, 1])
        assert isinstance(model, GLM)
        assert isinstance(model, BaseEstimator)
        assert not model._fit_done
        assert model._family.mean(np.zeros((2, 1))).shape == (2, 1)
