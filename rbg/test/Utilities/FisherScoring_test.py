"""Known-solution test for Fisher scoring with an intercept-only logistic model."""
import numpy as np

from rbg.Utilities import ExponentialFamily, FisherScoring


def test_logistic_intercept_only() -> None:
    """Recover log(4/6) and final Fisher information 2.4 from ten responses."""
    X = np.ones((10, 1))
    y = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def cumulant(theta: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, theta[:, 0])

    def mean(theta: np.ndarray) -> np.ndarray:
        return np.exp(-np.logaddexp(0.0, -theta))

    def curvature(theta: np.ndarray) -> np.ndarray:
        probability = mean(theta)
        return (probability * (1.0 - probability))[:, :, None]

    def inverse_mean(mu: np.ndarray) -> np.ndarray:
        return np.log(mu) - np.log1p(-mu)

    family = ExponentialFamily(
        b=cumulant,
        b_prime=mean,
        b_second=curvature,
        h_inverse=inverse_mean,
        phi=1.0,
    )
    beta, fisher = FisherScoring.solve(
        X, y, family, beta0=np.zeros(1), tol=1e-10, max_iter=100,
    )

    assert isinstance(beta, np.ndarray)
    assert beta.shape == (1,)
    assert isinstance(fisher, np.ndarray)
    assert fisher.shape == (1, 1)
    np.testing.assert_allclose(beta, [np.log(0.4 / 0.6)], rtol=0, atol=1e-8)
    np.testing.assert_allclose(fisher, [[2.4]], rtol=0, atol=1e-8)


def test_poisson_backtracking_rejects_overflowing_trial() -> None:
    """A full first step overflows, but backtracking recovers log(mean(y))."""
    X = np.ones((4, 1))
    y = np.array([900.0, 1000.0, 1100.0, 1000.0])
    family = ExponentialFamily(
        b=lambda theta: np.exp(theta[:, 0]),
        b_prime=np.exp,
        b_second=lambda theta: np.exp(theta)[:, :, None],
        h_inverse=np.log,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        beta, fisher = FisherScoring.solve(
            X, y, family, beta0=np.zeros(1), tol=1e-8, max_iter=100,
        )

    np.testing.assert_allclose(beta, [np.log(y.mean())], atol=1e-10, rtol=0)
    np.testing.assert_allclose(fisher, [[y.sum()]], atol=1e-6, rtol=0)
