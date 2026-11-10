import warnings
import numpy as np

class ExponentialFamily():
    """Represent an exponential family through its cumulant and derivatives.

    Inputs use one row per observation and one column per response component.
    For scalar responses, retain a single component column. Family functions
    must support these batched inputs and correspond to a valid distribution.
    """

    def __init__(self, b, b_prime, b_second, h_inverse, phi=1.0):
        """Store the family functions and dispersion parameter.

        Args:
            b: Cumulant function returning an array of shape (n_samples,).
            b_prime: Gradient returning means of shape (n_samples, n_components).
            b_second: Hessian returning arrays of shape
                (n_samples, n_components, n_components).
            h_inverse: Inverse of b_prime on its mean domain. With a canonical
                link, map response means to natural parameters and predictors.
            phi: Finite, positive dispersion parameter; fixed at 1 for the
                standard categorical family.

        Raises:
            ValueError: If a family function is not callable or phi is invalid.
        """
        if not all(callable(f) for f in (b, b_prime, b_second, h_inverse)):
            raise ValueError("b, b_prime, b_second, and h_inverse must be callable.")
        if not isinstance(phi, (int, float, np.integer, np.floating)) or not np.isfinite(phi) or phi <= 0:
            raise ValueError("phi must be finite and positive.")
        self.b = b
        self.b_prime = b_prime
        self.b_second = b_second
        self.h_inverse = h_inverse
        self._phi = phi

    def log_likelihood(self, y, theta, weights=1.0):
        """Return the parameter-dependent part of the summed log-likelihood.

        Compute sum(weights / phi * (inner_product(y, theta) - b(theta))).
        The additive normalization term is omitted. This is the full
        log-likelihood only when that term is zero.

        Args:
            y: Response or sufficient-statistic matrix of shape
                (n_samples, n_components).
            theta: Natural-parameter matrix with the same shape as y.
            weights: Scalar weight or array of shape (n_samples,).

        Returns:
            Sum of the parameter-dependent log-likelihood contributions.

        Raises:
            ValueError: If input dimensions or weight dimensions are incompatible.
        """
        y = np.asarray(y)
        theta = np.asarray(theta)
        weights = np.asarray(weights)
        if y.ndim != 2 or theta.ndim != 2 or y.shape != theta.shape or y.size == 0:
            raise ValueError("y and theta must be nonempty matrices with the same shape.")
        if weights.ndim != 0 and weights.shape != (theta.shape[0],):
            raise ValueError("weights must be a scalar or one value per observation.")
        b_values = np.asarray(self.b(theta))
        terms = (weights / self._phi) * (np.sum(y*theta, axis=1) - b_values)
        return np.sum(terms)

    def mean(self,theta):
        """Evaluate the mean response as the gradient of the cumulant.

        Args:
            theta: Nonempty matrix of shape (n_samples, n_components).

        Returns:
            Mean responses returned by b_prime, expected to match theta's shape.

        Raises:
            ValueError: If theta is not a nonempty two-dimensional array.
        """
        theta = np.asarray(theta)
        if theta.ndim != 2 or theta.size == 0:
            raise ValueError("theta must be a nonempty matrix.")
        result = np.asarray(self.b_prime(theta))

        return result

    def covariance(self,theta, weights = 1.0):
        """Evaluate conditional response covariance as phi / weights * b_second.

        These are response covariance matrices, not coefficient covariances.

        Args:
            theta: Nonempty matrix of shape (n_samples, n_components).
            weights: Scalar weight or array of shape (n_samples,). The caller
                must supply positive weights.

        Returns:
            Array of shape (n_samples, n_components, n_components), assuming
            b_second returns a Hessian per observation.

        Raises:
            ValueError: If theta or weight dimensions are incompatible.
        """
        theta = np.asarray(theta)
        weights = np.asarray(weights)
        if theta.ndim != 2 or theta.size == 0:
            raise ValueError("theta must be a nonempty matrix.")
        if weights.ndim != 0 and weights.shape != (theta.shape[0],):
            raise ValueError("weights must be a scalar or one value per observation.")
        result = np.asarray(self.b_second(theta))
        scale = self._phi / weights
        if weights.ndim == 1:
            scale = scale[:, None, None]
        return scale * result

