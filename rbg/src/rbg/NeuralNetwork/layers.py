"""Layers and the Parameter class."""

from abc import ABC, abstractmethod

import numpy as np


class Parameter:
    """A trainable array and its gradient.

    Attributes:
        data: The values, updated in place by the optimizer.
        grad: dL/d(data) from the last backward, same shape as data.
        name: Short name inside its layer, like "W" or "b".
    """

    def __init__(self, data: np.ndarray, name: str = "") -> None:
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter(name={self.name!r}, shape={self.data.shape})"


class Layer(ABC):
    """Base class of every layer.

    forward(x) receives the input batch and returns the output. backward(grad)
    receives dL/d(output), writes the gradient of the layer's own parameters
    (overwriting the old one) and returns dL/d(input). Use check_layer to test
    the backward of a new layer.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

    def parameters(self) -> list[Parameter]:
        """Return the trainable parameters of the layer ([] if it has none)."""
        return []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Dense(Layer):
    """Fully connected layer: y = x @ W + b.

    W has shape (n_in, n_out) and starts with He initialization (normal with
    std sqrt(2 / n_in)). b has shape (n_out,) and starts at zero.
    """

    def __init__(self, n_in: int, n_out: int, random_state: int | np.random.Generator | None = None) -> None:
        """Create the layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            random_state: Seed or generator for the initial weights. Pass the
                same generator to every layer of a network, not the same int,
                or layers of equal shape start with equal weights.
        """
        rng = np.random.default_rng(random_state)
        self.n_in = n_in
        self.n_out = n_out
        self.W = Parameter(rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)), "W")
        self.b = Parameter(np.zeros(n_out), "b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the layer to a batch of shape (N, n_in), returning (N, n_out)."""
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"Dense({self.n_in}, {self.n_out}) expected input of shape (N, {self.n_in}), got {x.shape}.")
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Take dL/d(output) of shape (N, n_out) and return dL/dx, shape (N, n_in)."""
        self.W.grad = self.x.T @ grad
        self.b.grad = grad.sum(axis=0)
        return grad @ self.W.data.T

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"Dense({self.n_in}, {self.n_out})"
