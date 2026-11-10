"""Loss functions used for training.

forward(prediction, target) returns the mean loss over the batch, and
backward() returns dL/d(prediction) for the last forward. This is the only
place where the 1/N of the batch mean appears.

These are not the metrics in rbg.Utilities: a loss also needs a backward.
"""

import numpy as np


class MSELoss:
    """Mean squared error, for regression.

    L = mean of (prediction - target)^2 over all elements. prediction and
    target must have the same shape, for example (N, 1).
    """

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        if prediction.shape != target.shape:
            raise ValueError(f"MSELoss: prediction has shape {prediction.shape} but target has shape {target.shape}.")
        diff = prediction - target
        self.grad = 2 * diff / diff.size
        return float((diff ** 2).mean())

    def backward(self) -> np.ndarray:
        return self.grad

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)


class CrossEntropyLoss:
    """Softmax followed by cross-entropy, for classification.

    Takes logits of shape (N, K), the raw output of the last Dense layer, and
    labels of shape (N,), integers in [0, K). L is the batch mean of
    -log p[i, labels[i]] with p = softmax(logits). Doing both steps together
    keeps L finite even for logits like [[1000, 0, -1000]].
    """

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        labels = np.asarray(labels)
        if logits.ndim != 2 or labels.shape != (logits.shape[0],):
            raise ValueError(f"CrossEntropyLoss: expected logits (N, K) and labels (N,), got {logits.shape} and {labels.shape}.")
        rows = np.arange(logits.shape[0])

        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))

        self.p = np.exp(log_p)
        self.labels = labels
        return float(-log_p[rows, labels].mean())

    def backward(self) -> np.ndarray:
        n = self.p.shape[0]
        grad = self.p.copy()
        grad[np.arange(n), self.labels] -= 1
        return grad / n

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> float:
        return self.forward(logits, labels)
