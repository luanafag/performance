import numpy as np
import pytest

from rbg.NeuralNetwork import Dense, Layer, ReLU, Sigmoid, Tanh, check_layer, softmax

LAYERS = [
    lambda rng: Dense(4, 3, rng),
    lambda rng: ReLU(),
    lambda rng: Sigmoid(),
    lambda rng: Tanh(),
]


def test_dense_shapes():
    layer = Dense(4, 3, random_state=0)
    assert layer.W.data.shape == (4, 3)
    assert layer.b.data.shape == (3,)
    assert layer.forward(np.ones((5, 4))).shape == (5, 3)


def test_dense_wrong_input():
    with pytest.raises(ValueError):
        Dense(4, 3).forward(np.ones((5, 2)))


def test_dense_same_seed_same_weights():
    np.testing.assert_array_equal(Dense(4, 3, random_state=7).W.data, Dense(4, 3, random_state=7).W.data)


def test_layer_needs_forward_and_backward():
    class OnlyForward(Layer):
        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        OnlyForward()


@pytest.mark.parametrize("make_layer", LAYERS)
def test_forward_does_not_change_input(make_layer):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4))
    original = x.copy()
    make_layer(rng).forward(x)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("make_layer", LAYERS)
def test_layer_gradients(make_layer):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 4))
    for name, error in check_layer(make_layer(rng), x, random_state=rng).items():
        assert error < 1e-6, f"gradient of {name}"


def test_backward_overwrites():
    rng = np.random.default_rng(3)
    layer = Dense(4, 3, rng)
    x = rng.normal(size=(5, 4))
    g = rng.normal(size=(5, 3))
    layer.forward(x)
    layer.backward(g)
    first = layer.W.grad.copy()
    layer.forward(x)
    layer.backward(g)
    np.testing.assert_allclose(layer.W.grad, first)


def test_sigmoid_extreme():
    out = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])


def test_softmax():
    p = softmax(np.array([[1000.0, 1000.0], [0.0, np.log(3)]]))
    np.testing.assert_allclose(p, [[0.5, 0.5], [0.25, 0.75]])
