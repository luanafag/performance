"""Activation functions, as layers without parameters.

There is no Softmax layer: a classifier ends in a Dense layer that returns
logits, and the softmax lives inside CrossEntropyLoss. The softmax function
here turns logits into probabilities when you want to look at them.
"""

import numpy as np

from .layers import Layer


class ReLU(Layer):
    """relu(x) = max(0, x), element-wise."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.x > 0)


class Sigmoid(Layer):
    """sigmoid(x) = 1 / (1 + exp(-x)), element-wise."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Written with tanh so that x = +-1000 does not overflow.
        self.s = (np.tanh(x / 2) + 1) / 2
        return self.s

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.s * (1 - self.s)


class Tanh(Layer):
    """tanh(x), element-wise."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.t = np.tanh(x)
        return self.t

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self.t ** 2)


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn logits into probabilities.

    Args:
        z: Logits of shape (N, K).

    Returns:
        Probabilities of shape (N, K); each row sums to 1.
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
