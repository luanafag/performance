import numpy as np
from rbg.Classification.classifier import Classifier

class KNN(Classifier):
    """K-nearest neighbors classifier for categorical labels."""

    def __init__(self, labels=None, k=3, reference_class = None):
        """Initialize the classifier with class labels and the number of neighbors.

        Args:
            labels: Unique class labels supported by the classifier.
            k: Number of nearest neighbors used to classify each observation.
                Defaults to 3.
            reference_class: Reference label used to initialize the base
                classifier.

        Raises:
            ValueError: If the class labels or reference label are invalid,
                as defined by the base classifier.
        """
        Classifier.__init__(self, labels, reference_class)
        self.set_params(k)

    def set_params(self, k):
        """Set the number of nearest neighbors.

        Args:
            k: Positive integer specifying the number of neighbors to consider.

        Returns:
            The classifier instance, allowing method chaining.

        Raises:
            ValueError: If k is not a positive integer or is a boolean.
        """
        if isinstance(k, bool) == False:
            if isinstance(k, int) and k>0:
                self.k = k
            else:
                raise ValueError("k must be a positive integer.")
        else:
            raise ValueError("k must be a positive integer.")
        return self

    def get_params(self):
        """Return the current classifier parameters.

        Returns:
            Dictionary containing k, the number of nearest neighbors.
        """
        return {"k": self.k}

    def fit(self, X,y):
        """Store the training data after validating their format and contents.

        Args:
            X: Numerical training feature matrix of shape
                (n_samples, n_features). All values must be finite.
            y: One-dimensional array of class labels of shape (n_samples,).
                All labels must belong to the set declared at initialization.

        Raises:
            ValueError: If X or y has an invalid dimension, X contains
                non-numerical or non-finite values, y contains unknown labels,
                the numbers of observations do not match, the data are empty,
                or X has no features.
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
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match.")
        if X.shape[0] == 0 or y.shape[0] == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        if self._labels is None:
            self._set_labels(np.unique(y))
        elif not np.isin(y, self._labels).all():
            raise ValueError("y contains unknown class labels.")

        self._X_train = X
        self._y_train = y
        self._fit_done = True

    def predict(self, X):
        """Predict class labels using the majority vote of the nearest neighbors.

        For each observation, calculate the Euclidean distances to the training
        observations, select the k nearest neighbors, and assign the most
        frequent class among their labels.

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
        if self._fit_done == False:
            raise ValueError("The model must be trained before prediction.")

        X = np.asarray(X)

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

        def min_distance(x, k):
            """Return the indices of the k training observations closest to x.

            Args:
                x: Feature vector of a single observation.
                k: Number of nearest neighbors to select.

            Returns:
                Indices of the k training observations with the smallest
                Euclidean distances to x.
            """
            dist = [sum((x[i]-amostra[i])**2 for i in range(len(x)))**(1/2) for amostra in self._X_train]
            indice = np.argsort(dist)[:k]
            return indice

        def class_neighbors(indice):
            """Retrieve the class labels of the selected neighbors.

            Args:
                indice: Indices of the selected training observations.

            Returns:
                Array of class labels corresponding to the selected neighbors.
            """
            neighbors = self._y_train[indice]
            return neighbors

        prediction = [class_neighbors(min_distance(x, self.k)) for x in X]

        def final_class(prediction):
            """Determine the most frequent class among the neighbors.

            Args:
                prediction: Array of class labels assigned to the neighbors.

            Returns:
                Class label with the highest frequency. Ties are resolved by
                selecting the first class returned by np.unique.
            """
            labels, counts = np.unique(prediction, return_counts=True)
            return labels[np.argmax(counts)]

        predicted_labels = np.array([final_class(neighbors) for neighbors in prediction])
        return predicted_labels
