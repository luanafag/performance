"""Multilayer perceptrons with the usual rbg interface.

MLPClassifier and MLPRegressor build a Sequential network of Dense layers
inside fit, so they work with cv_score, GridSearch and RandomizedSearch.
"""

import numpy as np

from rbg.Classification import Classifier
from rbg.Preprocessing import check_array

from .activations import ReLU, Sigmoid, Tanh, softmax
from .layers import Dense
from .losses import CrossEntropyLoss, MSELoss
from .model import Sequential
from .optimizers import SGD, Adam

ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
OPTIMIZERS = {"adam": Adam, "sgd": SGD}


class _MLP:
    """Hyperparameters and training code shared by MLPClassifier and MLPRegressor."""

    _param_names = ("hidden_layers", "activation", "optimizer", "learning_rate", "epochs", "batch_size", "random_state")

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64,),
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.network: Sequential | None = None
        self.history: dict[str, list[float]] | None = None
        self._fit_done = False

    def get_params(self) -> dict:
        """Return the hyperparameters as a dict."""
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, **params):
        """Change the given hyperparameters. A previous fit is discarded.

        Raises:
            ValueError: If a parameter name is unknown.
        """
        for name, value in params.items():
            if name not in self._param_names:
                raise ValueError(f"Unknown parameter '{name}'.")
            setattr(self, name, tuple(value) if name == "hidden_layers" else value)
        if params:
            self._fit_done = False
        return self

    def _train(self, X: np.ndarray, y: np.ndarray, n_out: int, loss: MSELoss | CrossEntropyLoss) -> None:
        """Build a new network for X and train it on (X, y)."""
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got '{self.activation}'.")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}, got '{self.optimizer}'.")

        # One generator for everything, so each layer gets different weights.
        rng = np.random.default_rng(self.random_state)
        sizes = [X.shape[1], *self.hidden_layers]
        layers = []
        for n_in, n_hidden in zip(sizes[:-1], sizes[1:]):
            layers += [Dense(n_in, n_hidden, rng), ACTIVATIONS[self.activation]()]
        layers.append(Dense(sizes[-1], n_out, rng))

        self.network = Sequential(layers)
        optimizer = OPTIMIZERS[self.optimizer](self.network.parameters(), lr=self.learning_rate)
        self.history = self.network.fit(X, y, loss, optimizer, self.epochs, self.batch_size, random_state=rng, verbose=False)
        self._fit_done = True

    def _output(self, X: np.ndarray) -> np.ndarray:
        """Return the raw network output for new data."""
        if not self._fit_done:
            raise RuntimeError("The model has not been fitted yet.")
        X = check_array(X)
        if X.shape[1] != self.network.layers[0].n_in:
            raise ValueError("X must have the same number of features as the training data.")
        return self.network.predict(X)


class MLPClassifier(_MLP, Classifier):
    """Neural network classifier trained with softmax cross-entropy.

    Labels can be anything np.unique can sort (ints, strings, ...). Output
    unit k of the network stands for self.labels[k].
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64,),
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int | np.random.Generator | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        """Set the hyperparameters. Nothing is trained until fit.

        Args:
            hidden_layers: Number of units of each hidden layer.
            activation: "relu", "tanh" or "sigmoid".
            optimizer: "adam" or "sgd".
            learning_rate: Step size of the optimizer.
            epochs: Number of passes over the training data.
            batch_size: Examples per training step.
            random_state: Seed or generator for the weights and the shuffling.
            labels: All the class labels. If None, they are taken from y in
                fit. Give them when a training set can miss some class, as in
                a cross-validation fold.
        """
        _MLP.__init__(self, hidden_layers, activation, optimizer, learning_rate, epochs, batch_size, random_state)
        Classifier.__init__(self, labels)
        self._given_labels = labels

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train a new network on (X, y).

        Args:
            X: Features of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            The fitted model.

        Raises:
            ValueError: If the shapes do not match, there are fewer than two
                classes, or y has a label that is not in labels.
        """
        X = check_array(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per row of X.")
        self._set_labels(self._given_labels if self._given_labels is not None else np.unique(y))
        if len(self.labels) < 2:
            raise ValueError("At least two classes are required.")
        if not np.isin(y, self.labels).all():
            raise ValueError("y contains labels that are not in labels.")

        y_index = np.argmax(y[:, None] == self.labels, axis=1)
        self._train(X, y_index, len(self.labels), CrossEntropyLoss())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, n_classes), columns in self.labels order."""
        return softmax(self._output(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely label of each row of X."""
        return self.labels[self._output(X).argmax(axis=1)]


class MLPRegressor(_MLP):
    """Neural network regressor trained with the mean squared error."""

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64,),
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Set the hyperparameters. Nothing is trained until fit.

        Args:
            hidden_layers: Number of units of each hidden layer.
            activation: "relu", "tanh" or "sigmoid".
            optimizer: "adam" or "sgd".
            learning_rate: Step size of the optimizer.
            epochs: Number of passes over the training data.
            batch_size: Examples per training step.
            random_state: Seed or generator for the weights and the shuffling.
        """
        super().__init__(hidden_layers, activation, optimizer, learning_rate, epochs, batch_size, random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Train a new network on (X, y).

        Args:
            X: Features of shape (n_samples, n_features).
            y: Targets of shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            The fitted model.

        Raises:
            ValueError: If X and y do not have the same number of rows.
        """
        X = check_array(X)
        y = np.asarray(y, dtype=float)
        if y.ndim not in (1, 2) or len(y) != len(X):
            raise ValueError("y must have shape (n_samples,) or (n_samples, n_outputs), matching X.")
        self._y_is_vector = y.ndim == 1
        Y = y.reshape(len(y), -1)
        self._train(X, Y, Y.shape[1], MSELoss())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions with the same number of columns as the training y."""
        out = self._output(X)
        return out[:, 0] if self._y_is_vector else out
