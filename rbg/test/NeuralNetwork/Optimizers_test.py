import numpy as np

from rbg.NeuralNetwork import SGD, Adam, Parameter


def test_sgd_updates_in_place():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -1.0])
    array_before = p.data
    SGD([p], lr=0.1).step()
    assert p.data is array_before
    np.testing.assert_allclose(p.data, [0.95, 2.1])


def test_adam_first_step():
    # With bias correction the first step has size lr * sign(grad).
    p = Parameter(np.array([1.0, 1.0]))
    p.grad = np.array([3.0, -0.01])
    Adam([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [0.9, 1.1], atol=1e-6)


def test_adam_minimizes_quadratic():
    p = Parameter(np.array([0.0]))
    optimizer = Adam([p], lr=0.05)
    for _ in range(1000):
        p.grad = 2 * (p.data + 3)
        optimizer.step()
    np.testing.assert_allclose(p.data, [-3.0], atol=1e-3)
