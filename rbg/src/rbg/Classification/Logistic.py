import numpy as np
from rbg.Regression import GLM
from rbg.Utilities import ExponentialFamily
from .classifier import Classifier

class Logistic(Classifier, GLM):
    """Multinomial logistic classifier using a reference-class GLM.

    Estimate K-1 coefficient vectors, fixing the reference-class coefficients
    to zero. Probability columns follow self._labels, with the reference last.
    Requires a GLM and Fisher-scoring solver supporting multivariate responses.
    """

    def __init__(self, classes=None, reference_class=None, fit_intercept=True, inference=False, *, labels=None):
        """Configure labels and initialize the categorical GLM.

        Args:
            labels: Compatibility alias for classes.
            classes: Optional unique class labels. If None, they are identified
                automatically during fitting.
            reference_class: Reference label, defaulting to the first supplied.
            fit_intercept: Whether to estimate an intercept for each modeled class.
            inference: Whether the GLM computes coefficient covariance during fit.

        Raises:
            ValueError: If class labels or the reference label are invalid.
        """
        if labels is not None:
            if classes is not None:
                raise ValueError("Provide only one of classes or labels.")
            classes = labels
        Classifier.__init__(self, classes, reference_class)
        self._fit_intercept = fit_intercept
        self._inference = inference
        self._glm_initialized = False

        if classes is not None:
            GLM.__init__(self, family=self._make_family(),fit_intercept=fit_intercept, inference=inference)
            self._glm_initialized = True

    def fit(self, X,y):
        """Encode class labels and delegate coefficient estimation to the GLM.

        Args:
            X: Feature matrix of shape (n_samples, p_features), without an
                automatically generated intercept column.
            y: Original class labels of shape (n_samples,).

        Returns:
            Coefficient matrix of shape (p, K-1), where p includes the intercept when enabled.

        Raises:
            ValueError: If y is not one-dimensional, contains unknown labels,
            or training inputs fail GLM validation.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")
        observed_labels = np.unique(y)
        if self._labels is None or not np.isin(observed_labels, self.labels).all():
            labels = observed_labels if self.labels is None else np.unique(np.concatenate((self.labels, observed_labels)))
            self._set_labels(labels, self._reference_class)
            self._glm_initialized = False
        if len(self._labels) < 2:
            raise ValueError("At least two classes are required.")
        if not self._glm_initialized:
            GLM.__init__(self, family=self._make_family(),fit_intercept=self._fit_intercept,inference=self._inference)
            self._glm_initialized = True
        if not np.isin(y, self.labels).all():
            raise ValueError("y contains unknown class labels.")

        Y = (y[:,None] == self._labels[:-1][None,:]).astype(float)
        return GLM.fit(self, X, Y)

    def predict_proba(self, X):
        """Return probabilities for all classes, including the reference.

        Args:
            X: Matrix of shape (n_new_samples, p_features), in training order.

        Returns:
            Probability matrix of shape (n_new_samples, K). Columns correspond
            to self._labels; the final column is the reference probability.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has an incompatible shape.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        pi = GLM.predict(self, X)
        pi_reference = 1 - pi.sum(axis=1, keepdims=True)
        return np.column_stack((pi, pi_reference))

    def predict(self, X):
        """Return the label with the highest probability for each observation.

        Args:
            X: Feature matrix of shape (n_new_samples, p_features).

        Returns:
            Original class labels of shape (n_new_samples,). Ties select the
            first maximum in the stored class order.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has an incompatible shape.
        """
        pi = self.predict_proba(X)
        return self._labels[np.argmax(pi, axis=1)]

    def k_function(self, k):
        """Return the fitted probability function for class k."""
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        matches = np.flatnonzero(self._labels == k)
        if not matches.size:
            raise ValueError("Unknown class label.")
        column = int(matches[0])

        def probability(X):
            X = np.asarray(X)
            single = X.ndim == 1
            result = self.predict_proba(X[None, :] if single else X)[:, column]
            return float(result[0]) if single else result

        return probability

    def k_coef(self, k):
        """Return class coefficients, or zeros for the reference class."""
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        matches = np.flatnonzero(self._labels == k)
        if not matches.size:
            raise ValueError("Unknown class label.")
        column = int(matches[0])
        if column == len(self._labels) - 1:
            return np.zeros(self._beta.shape[0])
        return self._beta[:, column].copy()

    inference = GLM.inference
    
    def _make_family(self):
        """Build the categorical exponential family with dispersion fixed at one.

        Returns:
            ExponentialFamily configured with the cumulant, gradient, Hessian,
            and inverse mean mapping for K-1 non-reference components.
        """
        self._K = len(self._labels)
        def b(theta):
            """Return log(1 + sum(exp(theta))) per row of an (n, K-1) matrix."""
            return np.logaddexp.reduce(
                np.column_stack((np.zeros(len(theta)), theta)), axis=1
            )
        def b_prime(theta):
            """Return the (n, K-1) probabilities of the non-reference classes."""
            pi = np.exp(theta - b(theta)[:, None])
            return pi
        def b_second(theta):
            """Return Hessians diag(pi) - pi*pi.T, with shape (n, K-1, K-1)."""
            pi = b_prime(theta)
            n, n_components = pi.shape
            cov_matrix = np.zeros((n, n_components, n_components))
            for i in range(n):
                cov_matrix[i] = np.diag(pi[i]) - np.outer(pi[i], pi[i])
            return cov_matrix
        def h_inverse(mu):
            """Return log-odds relative to the reference for each row of mu.

            All supplied probabilities and the remaining reference probability
            must be strictly positive. The caller must satisfy this domain.
            """
            pi = mu
            pi_reference = 1 - np.sum(pi, axis=1, keepdims=True)
            theta = np.log(pi) - np.log(pi_reference)
            return theta

        return ExponentialFamily(b,b_prime,b_second,h_inverse)