import numpy as np
from rbg.Classification import Classifier

class LDA(Classifier):
    """Linear discriminant analysis classifier for categorical labels.

    Assumes that all classes share a common covariance matrix and uses
    class-specific means and prior probabilities to discriminate between
    classes.
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
        """Estimate class means and the shared covariance matrix from training data.

        Args:
            X: Numerical training feature matrix of shape
                (n_samples, n_features). All values must be finite.
            y: One-dimensional array of class labels of shape (n_samples,).
                All labels must belong to the set declared at initialization.

        Raises:
            ValueError: If X or y has an invalid dimension, X contains
                non-numerical or non-finite values, y contains unknown labels,
                the numbers of observations do not match, X is empty or has no
                features, fewer than two classes are present, there are not
                enough samples to estimate the covariance matrix, or the
                estimated covariance matrix is singular.
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
        if X.shape[0] <= self.labels.size:
            raise ValueError("There must be more samples to estimate covariance.")

        self._X_train = X
        self._y_train = y

        self._covariance = self.cov_matrix()
        if np.linalg.matrix_rank(self._covariance) < self._covariance.shape[0]:
            raise ValueError("The covariance matrix is singular.")

        self._fit_done = True

    def mean_array(self, k):
        """Calculate the mean feature vector for a given class.

        Args:
            k: Class label for which the mean is calculated.

        Returns:
            Mean feature vector of shape (n_features,) for class k.

        Raises:
            ValueError: If the specified class has no training samples.
        """
        x_k = self._X_train[self._y_train == k]
        if x_k.shape[0] == 0:
            raise ValueError("The class must have training samples.")
        mean_k = np.mean(x_k, axis = 0)
        return mean_k

    def cov_matrix(self):
        """Estimate the pooled covariance matrix across all training classes.

        Computes the sum of the within-class scatter matrices divided by the
        total number of observations minus the number of observed classes.

        Returns:
            Pooled covariance matrix of shape (n_features, n_features).
        """
        matriz = np.zeros((self._X_train.shape[1], self._X_train.shape[1]))
        detours = [self._X_train[self._y_train == k] - self.mean_array(k) for k in self.labels]
        products = [detour.T @ detour for detour in detours]
        matriz = np.sum(np.stack(products), axis=0)/(self._y_train.size - self.labels.size)
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
        """Construct the linear discriminant function for a given class.

        The discriminant combines the class mean, the inverse of the shared
        covariance matrix, and the logarithm of the empirical class prior.

        Args:
            k: Class label for which the discriminant function is constructed.

        Returns:
            Callable that evaluates the discriminant function for a feature
            vector x.
        """
        def function_discriminant(x):
            """Evaluate the linear discriminant score for a feature vector.

            Args:
                x: Feature vector of shape (n_features,).

            Returns:
                Discriminant score for class k.
            """
            function = (x.T @ np.linalg.inv(self._covariance) @ self.mean_array(k) 
                    - 1/2 * self.mean_array(k).T @ np.linalg.inv(self._covariance)
                    @ self.mean_array(k) + np.log(self.prior(k)))
            return function
        return function_discriminant

    def predict(self, X):
        """Predict class labels by maximizing the linear discriminant scores.

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
