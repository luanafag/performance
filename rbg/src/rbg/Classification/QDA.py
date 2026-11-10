import numpy as np
from rbg.Classification.classifier import Classifier

class QDA(Classifier):
    """Quadratic discriminant analysis classifier for categorical labels.

    Assumes that each class has its own covariance matrix and uses
    class-specific means, covariance matrices, and prior probabilities
    to discriminate between classes.
    """

    def __init__(self, labels, reference_class=None):
        """Initialize the classifier with the supported class labels.

        Args:
            labels: Unique class labels supported by the classifier.
            reference_class: Reference label used to initialize the base
                classifier.
        """
        Classifier.__init__(self, labels, reference_class)

    def fit(self, X, y):
        """Estimate class means and covariance matrices from training data.

        Args:
            X: Numerical training feature matrix of shape
                (n_samples, n_features). All values must be finite.
            y: One-dimensional array of class labels of shape (n_samples,).
                All labels must belong to the set declared at initialization.

        Raises:
            ValueError: If X or y has an invalid dimension, X contains
                non-numerical or non-finite values, y contains unknown labels,
                the numbers of observations do not match, X is empty or has no
                features, fewer than two classes are present, any class has
                fewer than two training samples, or a class covariance matrix
                is singular.
        """
        self._fit_done = False

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if np.issubdtype(X.dtype, np.number) == False:
            raise ValueError("X must be a numerical array.")
        else:
            if np.isfinite(X).all() == False:
                raise ValueError("X must have only finite values.")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if not np.isin(y, self._labels).all():
            raise ValueError("y contains unknown class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match.")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")

        self.labels = np.unique(y)
        if self.labels.size < 2:
            raise ValueError("y must be at least two classes.")

        self._X_train = X
        self._y_train = y

        self._covariance = {}
        for k in self.labels:
            if np.sum(self._y_train == k) < 2:
                raise ValueError("Each class must have at least two samples.")
            self._covariance[k] = self.cov_matrix(k)
            if np.linalg.matrix_rank(self._covariance[k]) < self._covariance[k].shape[0]:
                raise ValueError("The covariance matrix is singular.")

        self._fit_done = True

    def mean_array(self, k):
        """Calculate the mean feature vector for a given class.

        Args:
            k: Class label for which the mean is calculated.

        Returns:
            Mean feature vector of shape (n_features,) for class k.
        """
        x_k = self._X_train[self._y_train == k]
        mean_k = np.mean(x_k, axis = 0)
        return mean_k

    def cov_matrix(self, k):
        """Estimate the sample covariance matrix for a given class.

        Computes the covariance matrix using the centered observations
        belonging to class k, divided by the number of observations in that
        class minus one.

        Args:
            k: Class label for which the covariance matrix is estimated.

        Returns:
            Covariance matrix of shape (n_features, n_features) for class k.
        """
        x_k = self._X_train[self._y_train == k]
        detours = x_k - self.mean_array(k)
        matriz = detours.T @ detours / (x_k.shape[0] - 1)
        return matriz

    def prior(self, k):
        """Calculate the empirical prior probability of a given class.

        Args:
            k: Class label for which the prior probability is calculated.

        Returns:
            Proportion of training observations belonging to class k.
        """
        sum_k = (self._y_train == k).sum()
        prior_k = sum_k/self._y_train.size
        return prior_k

    def k_functions(self, k):
        """Construct the quadratic discriminant function for a given class.

        The discriminant uses the class-specific covariance matrix and mean,
        the log-determinant of the covariance matrix, and the logarithm of the
        empirical class prior.

        Args:
            k: Class label for which the discriminant function is constructed.

        Returns:
            Callable that evaluates the quadratic discriminant score for
            a feature vector x.
        """
        def function_discriminant(x):
            """Evaluate the quadratic discriminant score for a feature vector.

            Args:
                x: Feature vector of shape (n_features,).

            Returns:
                Discriminant score for class k.

            Raises:
                ValueError: If the covariance matrix is not positive definite
                    according to its log-determinant sign.
            """
            covariance = self._covariance[k]
            mean = self.mean_array(k)
            difference = x - mean

            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("The covariance matrix is singular.")

            function = (-1/2 * logdet -1/2 * difference.T @ np.linalg.solve(covariance, difference)+ np.log(self.prior(k)))
            return function
        return function_discriminant

    def predict(self, X):
        """Predict class labels by maximizing the quadratic discriminant scores.

        For each observation, evaluates the discriminant function of every
        observed training class and selects the class with the highest score.

        Args:
            X: Numerical feature matrix of shape
                (n_new_samples, n_features). The number of features must match
                the training data, and all values must be finite.

        Returns:
            Array of predicted class labels of shape (n_new_samples,).

        Raises:
            ValueError: If the model has not been fitted, X has an invalid
                dimension, contains non-numerical or non-finite values, is
                empty, or has a different number of features from the
                training data.
        """
        X = np.asarray(X)
        if self._fit_done == False:
            raise ValueError("The model must be trained before prediction.")
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if np.issubdtype(X.dtype, np.number) == False:
            raise ValueError("X must be a numerical array.")
        else:
            if np.isfinite(X).all() == False:
                raise ValueError("X must have only finite values.")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must be the same number of features as the training X.") 

        functions_class = np.array([self.labels[np.argmax([self.k_functions(k)(x) for k in self.labels])]for x in X])
        return functions_class
