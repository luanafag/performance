"""Behavioral regression tests using real estimators and numerical calculations.

Lasso is imported inside its factory so a missing CoordinateDescent is reported
as a test error without preventing the Linear/Ridge tests from running.
"""

from importlib import import_module

import numpy as np
import pytest

from rbg.Utilities import Resample
from rbg.Regression import Linear
from rbg.Regression import Polynomial
from rbg.Regression import Ridge


class RegressionBehaviorChecks:
    """Shared checks, collected only through concrete pytest classes."""

    @pytest.fixture(autouse=True)
    def _subtest_context(self, subtests):
        self.subtests = subtests

    def setup_method(self) -> None:
        rng = np.random.default_rng(98)
        self.X = rng.normal(size=(40, 2))
        self.y = 2.0 + self.X @ np.array([1.5, -0.7]) + rng.normal(scale=0.2, size=40)
        self.X_second = rng.normal(loc=1.0, scale=2.0, size=(50, 2))
        self.y_second = -4.0 + self.X_second @ np.array([-0.8, 2.3]) + rng.normal(scale=0.5, size=50)

    def fit_model(self, model, X, y):
        if isinstance(model, (Linear, Polynomial)):
            return model.fit(X, y)
        return model.fit(X, y, alpha=0.3)

    def get_inference(self, model):
        if type(model).__name__ == "Lasso":
            return model.inference(n_bootstrap=8, seed=7)
        return model.inference()

    def test_refit_replaces_coefficients_residuals_and_inference(self) -> None:
        model = self.make_model()
        first_beta = self.fit_model(model, self.X, self.y).copy()
        first_inference = self.get_inference(model)
        first_covariance = first_inference[2].copy()
        first_residuals = first_inference[3].copy()

        second_beta = self.fit_model(model, self.X_second, self.y_second)
        second_inference = self.get_inference(model)
        fresh = self.make_model()
        expected_beta = self.fit_model(fresh, self.X_second, self.y_second)
        expected_inference = self.get_inference(fresh)

        assert not np.allclose(first_beta, second_beta)
        assert not np.allclose(first_covariance, second_inference[2])
        assert first_residuals.shape == (40,)
        assert second_inference[3].shape == (50,)
        np.testing.assert_allclose(second_beta, expected_beta, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(model.predict(self.X_second), fresh.predict(self.X_second))
        for observed, expected in zip(second_inference, expected_inference):
            np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(model.residuals(), self.y_second - model.predict(self.X_second))

    def test_predict_residuals_and_inference_require_fit(self) -> None:
        model = self.make_model()
        for method, args in ((model.predict, (self.X,)), (model.residuals, ()), (model.inference, ())):
            with self.subtests.test(method=method.__name__):
                with pytest.raises(RuntimeError, match="(?i)fit"):
                    method(*args)

    def test_fit_rejects_nonfinite_and_empty_data(self) -> None:
        cases = []
        for invalid_value in (np.nan, np.inf, -np.inf):
            X, y = self.X.copy(), self.y.copy()
            X[0, 0] = invalid_value
            cases.append((f"X contains {invalid_value}", X, self.y))
            y[0] = invalid_value
            cases.append((f"y contains {invalid_value}", self.X, y))
        cases.extend([
            ("no samples", np.empty((0, 2)), np.empty(0)),
            ("no features", np.empty((40, 0)), self.y),
            ("empty response", self.X, np.empty(0)),
        ])
        for label, X, y in cases:
            with self.subtests.test(case=label):
                with pytest.raises(ValueError):
                    self.fit_model(self.make_model(), X, y)

    def test_predict_rejects_nonfinite_data(self) -> None:
        model = self.make_model()
        self.fit_model(model, self.X, self.y)
        for value in (np.nan, np.inf, -np.inf):
            with self.subtests.test(value=value):
                X = self.X[:3].copy()
                X[0, 0] = value
                with pytest.raises(ValueError):
                    model.predict(X)


class TestLinearBehavior(RegressionBehaviorChecks):
    def make_model(self):
        return Linear()


class TestRidgeBehavior(RegressionBehaviorChecks):
    def make_model(self):
        return Ridge()

    def test_zero_alpha_matches_linear_even_with_dependent_columns(self) -> None:
        rng = np.random.default_rng(12)
        new_X = rng.normal(size=(7, 2))
        for intercept in (True, False):
            for dependent in (True, False):
                with self.subtests.test(intercept=intercept, dependent=dependent):
                    X, X_new = self.X, new_X
                    if dependent:
                        X = np.column_stack((X, 2 * X[:, 0] - 0.5 * X[:, 1]))
                        X_new = np.column_stack((X_new, 2 * X_new[:, 0] - 0.5 * X_new[:, 1]))
                    linear = Linear(fit_intercept=intercept)
                    ridge = Ridge(fit_intercept=intercept)
                    linear.fit(X, self.y)
                    ridge.fit(X, self.y, alpha=0.0)
                    np.testing.assert_allclose(ridge.predict(X), linear.predict(X), rtol=1e-9, atol=1e-10)
                    np.testing.assert_allclose(ridge.predict(X_new), linear.predict(X_new), rtol=1e-9, atol=1e-10)

    def test_rescaling_a_feature_preserves_predictions(self) -> None:
        for intercept in (True, False):
            with self.subtests.test(intercept=intercept):
                original, scaled = Ridge(fit_intercept=intercept), Ridge(fit_intercept=intercept)
                X_scaled = self.X.copy()
                X_scaled[:, 0] *= 37.0
                beta = original.fit(self.X, self.y, alpha=0.3)
                scaled_beta = scaled.fit(X_scaled, self.y, alpha=0.3)
                np.testing.assert_allclose(original.predict(self.X), scaled.predict(X_scaled), rtol=1e-8, atol=1e-9)
                expected_beta = beta.copy()
                expected_beta[int(intercept)] /= 37.0
                np.testing.assert_allclose(scaled_beta, expected_beta, rtol=1e-8, atol=1e-9)

    def test_invalid_penalties_are_rejected(self) -> None:
        for alpha in (-1.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subtests.test(alpha=alpha):
                with pytest.raises(ValueError):
                    Ridge().fit(self.X, self.y, alpha=alpha)


class TestLassoBehavior(RegressionBehaviorChecks):
    def setup_method(self) -> None:
        super().setup_method()
        from rbg.Regression.Lasso import Lasso

        self.lasso_class = Lasso

    def make_model(self, fit_intercept=True):
        return self.lasso_class(fit_intercept=fit_intercept)

    def test_high_penalty_zeros_feature_coefficients(self) -> None:
        for intercept in (True, False):
            with self.subtests.test(intercept=intercept):
                model = self.make_model(fit_intercept=intercept)
                beta = model.fit(self.X, self.y, alpha=1e6)
                np.testing.assert_allclose(beta[int(intercept):], 0.0, rtol=0, atol=1e-8)
                expected = self.y.mean() if intercept else 0.0
                if intercept:
                    assert beta[0] == pytest.approx(expected, rel=0, abs=5e-08)
                np.testing.assert_allclose(model.predict(self.X[:5]), np.full(5, expected), atol=1e-8)

    def test_rescaling_a_feature_preserves_predictions(self) -> None:
        for intercept in (True, False):
            with self.subtests.test(intercept=intercept):
                original = self.make_model(fit_intercept=intercept)
                scaled = self.make_model(fit_intercept=intercept)
                X_scaled = self.X.copy()
                X_scaled[:, 0] *= 37.0
                beta = original.fit(self.X, self.y, alpha=0.3)
                scaled_beta = scaled.fit(X_scaled, self.y, alpha=0.3)
                np.testing.assert_allclose(original.predict(self.X), scaled.predict(X_scaled), rtol=1e-6, atol=1e-7)
                expected_beta = beta.copy()
                expected_beta[int(intercept)] /= 37.0
                np.testing.assert_allclose(scaled_beta, expected_beta, rtol=1e-6, atol=1e-7)

    def test_seeded_inference_is_reproducible_across_models(self) -> None:
        first, second = self.make_model(), self.make_model()
        first.fit(self.X, self.y, alpha=0.3)
        second.fit(self.X, self.y, alpha=0.3)
        first_result = first.inference(n_bootstrap=12, seed=42)
        second_result = second.inference(n_bootstrap=12, seed=42)
        for observed, expected in zip(first_result, second_result):
            np.testing.assert_allclose(observed, expected, rtol=0, atol=0)

    def test_inference_caches_and_recomputes_for_new_options(self, monkeypatch) -> None:
        model = self.make_model()
        original_beta = model.fit(self.X, self.y, alpha=0.3).copy()
        # Record sampler calls while retaining the real sampling implementation.
        calls = []

        def record_resample(*args, **kwargs):
            calls.append(kwargs.copy())
            return Resample(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(import_module("rbg.Regression.Lasso"), "Resample", record_resample)
            first = model.inference(n_bootstrap=8, seed=42)
            cached = model.inference(n_bootstrap=8, seed=42)
            assert len(calls) == 1
            for observed, expected in zip(cached, first):
                np.testing.assert_array_equal(observed, expected)
            model.inference(n_bootstrap=8, seed=43)
            assert len(calls) == 2
            assert calls[-1]["random_state"] == 43
            model.inference(n_bootstrap=10, seed=43)
            assert len(calls) == 3
            assert calls[-1]["n_resamples"] == 10
        np.testing.assert_array_equal(model._beta, original_beta)

    def test_invalid_penalties_are_rejected(self) -> None:
        for alpha in (-1.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subtests.test(alpha=alpha):
                with pytest.raises(ValueError):
                    self.make_model().fit(self.X, self.y, alpha=alpha)


class TestPolynomialBehavior(RegressionBehaviorChecks):
    def make_model(self):
        return Polynomial([1, 1])

    def test_invalid_degrees_are_rejected(self) -> None:
        for degrees in ([], [0], [-1], [2, 0], [1.5], [[2]], [True], [np.nan], [np.inf]):
            with self.subtests.test(degrees=degrees):
                with pytest.raises(ValueError):
                    Polynomial(degrees)
