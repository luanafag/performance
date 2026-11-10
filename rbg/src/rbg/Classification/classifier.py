from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    """Abstract base class for classification models."""

    def __init__(self, labels=None, reference_class=None):
        """Store class labels and move the reference class to the last position.

        Args:
        labels: Optional nonempty one-dimensional sequence of unique class labels.
            If None, the class labels can be defined later during fitting.
        reference_class: Reference label. If None, use the first supplied label,
            then move it to the end of the stored class order.

        Raises:
            ValueError: If labels is empty, not one-dimensional, contains duplicates,
                or does not contain the reference label.
        """
        self._fit_done = False
        self._labels = None
        self.labels = None
        self._reference_class = reference_class

        if labels is not None:
            self._set_labels(labels, reference_class)

    def _set_labels(self, labels, reference_class=None):
        labels = np.asarray(labels)

        if labels.ndim != 1 or labels.size == 0:
           raise ValueError("labels must be a nonempty one-dimensional sequence.")
        if np.unique(labels).size != labels.size:
           raise ValueError("labels must contain unique labels.")
        if reference_class is None:
            reference_class = labels[0]
        if reference_class not in labels:
            raise ValueError("The reference class must be present in labels.")
        self.labels = labels.copy()
        self._reference_class = reference_class
        self._labels = np.concatenate((labels[labels != reference_class],labels[labels == reference_class]))

    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier using features X and original class labels y.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Class-label vector of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Return one original class label per input observation.

        Args:
            X: Feature matrix of shape (n_new_samples, p_features).

        Returns:
            Predicted label vector of shape (n_new_samples,).
        """
        pass
