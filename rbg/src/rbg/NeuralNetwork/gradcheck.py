"""Gradient check: compare a backward with a numerical derivative."""

from collections.abc import Callable

import numpy as np

from .layers import Layer


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central difference derivative of f with respect to each entry of x.

    Args:
        f: Function with no arguments that reads x and returns a float.
        x: Float array. Each entry is changed in place and restored afterwards.
        eps: Step of the central difference.

    Returns:
        Array with the shape of x.
    """
    grad = np.zeros(x.shape)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        f_plus = f()
        x[i] = old - eps
        f_minus = f()
        x[i] = old
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return max|a - b| / max(|a| + |b|).

    0 means equal. Below 1e-6 a backward is right; above 1e-3 it is wrong.
    """
    return float(np.abs(a - b).max() / max((np.abs(a) + np.abs(b)).max(), 1e-12))


def check_layer(layer: Layer, x: np.ndarray, random_state: int | np.random.Generator | None = None) -> dict[str, float]:
    """Compare layer.backward with the numerical gradient.

    Uses the loss L = sum(layer.forward(x) * G) with G random and fixed, so
    that dL/d(output) is G. Use float64 data.

    Args:
        layer: The layer to test.
        x: An input batch for the layer.
        random_state: Seed or generator for G.

    Returns:
        Relative error for the input ("x") and for each parameter, keyed by
        the parameter name, or "p0", "p1", ... when it has none.
    """
    rng = np.random.default_rng(random_state)
    x = np.array(x, dtype=float)

    out = layer.forward(x)
    G = rng.normal(size=out.shape)
    dx = layer.backward(G)

    def f() -> float:
        return float(np.sum(layer.forward(x) * G))

    errors = {"x": relative_error(dx, numerical_gradient(f, x))}
    for i, p in enumerate(layer.parameters()):
        errors[p.name or f"p{i}"] = relative_error(p.grad, numerical_gradient(f, p.data))
    return errors
