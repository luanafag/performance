"""Optimizers for training a network one batch at a time.

An optimizer receives the list of parameters when it is created, usually
model.parameters(). step() reads param.grad and updates param.data in place,
so the array a layer holds is the same before and after the step.

rbg.Utilities.AdamOptimizer is different: it minimizes a function of one
vector until convergence, which does not fit a training loop.
"""

from collections.abc import Iterable

import numpy as np

from .layers import Parameter


class SGD:
    """Gradient descent: p <- p - lr * grad."""

    def __init__(self, parameters: Iterable[Parameter], lr: float = 0.01) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.lr * p.grad


class Adam:
    """Adam (Kingma and Ba, 2014).

    Keeps a moving average of the gradient (momentum) and of the squared
    gradient (one step size per weight), both with bias correction.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
