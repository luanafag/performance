import numpy as np
import pytest

from rbg.NeuralNetwork import CrossEntropyLoss, MSELoss, numerical_gradient, relative_error, softmax
from rbg.Utilities import MSE


def test_mse_value_and_gradient():
    rng = np.random.default_rng(4)
    pred = rng.normal(size=(6, 2))
    target = rng.normal(size=(6, 2))
    loss = MSELoss()
    assert loss.forward(pred, target) == pytest.approx(MSE(pred, target))
    numeric = numerical_gradient(lambda: loss.forward(pred, target), pred)
    loss.forward(pred, target)
    assert relative_error(loss.backward(), numeric) < 1e-6


def test_mse_shape_mismatch():
    with pytest.raises(ValueError):
        MSELoss().forward(np.ones((4, 1)), np.ones(4))


def test_cross_entropy_value_and_gradient():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(6, 4))
    labels = rng.integers(0, 4, size=6)
    loss = CrossEntropyLoss()
    expected = -np.mean(np.log(softmax(logits)[np.arange(6), labels]))
    assert loss.forward(logits, labels) == pytest.approx(expected)
    numeric = numerical_gradient(lambda: loss.forward(logits, labels), logits)
    loss.forward(logits, labels)
    assert relative_error(loss.backward(), numeric) < 1e-6


def test_cross_entropy_extreme():
    value = CrossEntropyLoss().forward(np.array([[1000.0, 0.0, -1000.0]]), np.array([2]))
    assert np.isfinite(value)


def test_cross_entropy_wrong_labels():
    with pytest.raises(ValueError):
        CrossEntropyLoss().forward(np.zeros((4, 3)), np.zeros((4, 1), dtype=int))
