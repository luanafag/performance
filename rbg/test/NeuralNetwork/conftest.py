import numpy as np
import pytest


def make_spirals(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for k in range(n_classes):
        rows = slice(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X[rows] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        y[rows] = k
    return X, y


@pytest.fixture
def spirals():
    return make_spirals()


@pytest.fixture
def small_spirals():
    return make_spirals(n_per_class=10)
