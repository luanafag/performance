"""Dimensional contracts for exponential-family functions and weight broadcasting."""

import numpy as np
import pytest

from rbg.Utilities import ExponentialFamily


class TestExponentialFamilyDimensions:

    @pytest.fixture(autouse=True)
    def _subtest_context(self, subtests):
        self.subtests = subtests

    def setup_method(self) -> None:
        self.family = ExponentialFamily(
            b=lambda theta: np.sum(theta**2, axis=1) / 2,
            b_prime=lambda theta: theta.copy(),
            b_second=lambda theta: np.broadcast_to(
                np.eye(theta.shape[1]), (len(theta), theta.shape[1], theta.shape[1]),
            ).copy(),
            h_inverse=lambda mu: mu.copy(), phi=2.0,
        )

    def test_all_function_output_dimensions(self) -> None:
        for n, k in ((1, 1), (5, 1), (1, 3), (5, 3)):
            theta = np.arange(n * k, dtype=float).reshape(n, k) / 10
            for weights in (1.0, np.linspace(1.0, 2.0, n)):
                with self.subtests.test(samples=n, components=k, weight_shape=np.shape(weights)):
                    assert self.family.b(theta).shape == (n,)
                    assert self.family.b_prime(theta).shape == (n, k)
                    assert self.family.b_second(theta).shape == (n, k, k)
                    assert self.family.h_inverse(theta).shape == (n, k)
                    assert self.family.mean(theta).shape == (n, k)
                    covariance = self.family.covariance(theta, weights)
                    assert covariance.shape == (n, k, k)
                    likelihood = self.family.log_likelihood(theta, theta, weights)
                    assert np.ndim(likelihood) == 0
                    expected_covariance = np.stack([2 * np.eye(k) / w for w in np.broadcast_to(weights, (n,))])
                    np.testing.assert_allclose(covariance, expected_covariance)

    def test_mean_and_covariance_reject_nonmatrix_or_empty_inputs(self) -> None:
        for theta in (np.array(1.0), np.ones(3), np.ones((2, 1, 1)),
                      np.empty((0, 1)), np.empty((2, 0))):
            with self.subtests.test(theta=theta.shape):
                with pytest.raises(ValueError):
                    self.family.mean(theta)
                with pytest.raises(ValueError):
                    self.family.covariance(theta)

    def test_log_likelihood_rejects_response_dimension_mismatches(self) -> None:
        theta = np.ones((4, 2))
        for y in (np.ones(4), np.ones((3, 2)), np.ones((4, 1)), np.ones((4, 2, 1))):
            with self.subtests.test(y=y.shape):
                with pytest.raises(ValueError):
                    self.family.log_likelihood(y, theta)
        for y, natural in ((np.empty((0, 2)), np.empty((0, 2))),
                           (np.ones(4), np.ones(4))):
            with self.subtests.test(y=y.shape, theta=natural.shape):
                with pytest.raises(ValueError):
                    self.family.log_likelihood(y, natural)

    def test_covariance_and_likelihood_reject_wrong_weight_dimensions(self) -> None:
        theta = np.ones((4, 2))
        for weights in (np.ones(3), np.ones((4, 1)), np.ones((4, 2))):
            with self.subtests.test(weights=weights.shape):
                with pytest.raises(ValueError):
                    self.family.covariance(theta, weights)
                with pytest.raises(ValueError):
                    self.family.log_likelihood(theta, theta, weights)
