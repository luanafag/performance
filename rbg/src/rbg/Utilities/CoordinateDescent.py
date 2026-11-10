import warnings 
import numpy as np



class CoordinateDescent:
    """Coordinate descent for least squares with an L1 penalty (Lasso)."""

    @staticmethod
    def solve(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        fit_intercept: bool,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Minimize 0.5 * ||y - X @ beta||^2 + alpha * sum(|beta_j|).

        Args:
            X: Design matrix of shape (n, p).
            y: Target vector of shape (n,).
            alpha: Strength of the L1 penalty, must be nonnegative.
            fit_intercept: If True, column 0 of X is the intercept and is not penalized.
            tol: Stop when no coefficient changed more than tol in a full pass.
            max_iter: Maximum number of full passes over the coefficients.

        Returns:
            The coefficient vector, of shape (p,).
        """
        _, p = X.shape

        beta = np.zeros(p)
        residual = y.copy()  
        
        column_norms = np.zeros(p)
        for j in range(p):
            column_norms[j] = X[:, j] @ X[:, j]

        converged = False
        for _ in range(max_iter):
            biggest_change = 0.0

            for j in range(p):
                if column_norms[j] == 0.0:
                    continue

                old_value = beta[j]

                rho = X[:, j] @ residual + column_norms[j] * old_value

                if fit_intercept and j == 0:
                    new_value = rho / column_norms[j]
                else:
                    new_value = CoordinateDescent._soft_threshold(rho, alpha) / column_norms[j]

                difference = new_value - old_value
                residual = residual - difference * X[:, j]
                beta[j] = new_value

                if abs(difference) > biggest_change:
                    biggest_change = abs(difference)

            if biggest_change < tol:
                converged = True
                break

        if not converged:
            warnings.warn("CoordinateDescent did not converge.")

        return beta

    @staticmethod
    def _soft_threshold(z: float, threshold: float) -> float:
        """Move z towards zero by threshold, and return 0 if it would cross zero.

        Args:
            z: The value to shrink.
            threshold: How much to shrink, must be nonnegative.

        Returns:
            z - threshold if z > threshold, z + threshold if z < -threshold,and 0 otherwise.
        """
        if z > threshold:
            return z - threshold
        elif z < -threshold:
            return z + threshold
        else:
            return 0.0