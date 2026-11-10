import numpy as np
import pytest

from rbg.NeuralNetwork import (
    SGD,
    Adam,
    CrossEntropyLoss,
    Dense,
    MSELoss,
    ReLU,
    Sequential,
    Tanh,
    batches,
    confusion_matrix,
    numerical_gradient,
    relative_error,
)
from rbg.Utilities import Accuracy, train_test_split


def make_model(seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential([Dense(2, 32, rng), ReLU(), Dense(32, 3, rng)])


def test_batches_cover_everything():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    seen = []
    sizes = []
    for X_batch, y_batch in batches(X, y, 4, random_state=0):
        np.testing.assert_array_equal(X_batch[:, 0], y_batch)
        seen += list(y_batch)
        sizes.append(len(y_batch))
    assert sorted(seen) == list(range(10))
    assert sizes == [4, 4, 2]


def test_batches_no_shuffle():
    X = np.arange(5).reshape(5, 1)
    y = np.arange(5)
    assert [list(y_batch) for _, y_batch in batches(X, y, 2, shuffle=False)] == [[0, 1], [2, 3], [4]]


def test_batches_length_mismatch():
    with pytest.raises(ValueError):
        list(batches(np.zeros((5, 1)), np.zeros(4), 2))


def test_confusion_matrix():
    M = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
    np.testing.assert_array_equal(M, [[1, 0, 0], [0, 1, 0], [0, 1, 1]])


def test_parameters_and_names():
    model = make_model(0)
    assert len(model.parameters()) == 4
    assert [name for name, _ in model.named_parameters()] == ["0.W", "0.b", "2.W", "2.b"]


def test_model_gradient():
    rng = np.random.default_rng(1)
    model = Sequential([Dense(3, 5, rng), Tanh(), Dense(5, 2, rng)])
    x = rng.normal(size=(4, 3))
    labels = np.array([0, 1, 1, 0])
    loss = CrossEntropyLoss()
    loss.forward(model.forward(x), labels)
    model.backward(loss.backward())
    for p in model.parameters():
        numeric = numerical_gradient(lambda: loss.forward(model.forward(x), labels), p.data)
        assert relative_error(p.grad, numeric) < 1e-6


def test_fit_learns_spirals(spirals):
    X, y = spirals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = make_model(2)
    optimizer = Adam(model.parameters(), lr=0.02)
    history = model.fit(X_train, y_train, CrossEntropyLoss(), optimizer, epochs=200,
                        validation_data=(X_test, y_test), random_state=2, verbose=False)
    assert len(history["loss"]) == 200
    assert len(history["val_loss"]) == 200
    assert history["loss"][-1] < history["loss"][0]
    assert Accuracy(model.classify(X_train), y_train) > 0.9


def test_fit_regression():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = np.sin(3 * X)
    model = Sequential([Dense(1, 32, rng), Tanh(), Dense(32, 1, rng)])
    model.fit(X, y, MSELoss(), Adam(model.parameters(), lr=0.01), epochs=300, random_state=rng, verbose=False)
    assert model.evaluate(X, y, MSELoss()) < 0.01


def test_fit_is_reproducible(small_spirals):
    X, y = small_spirals
    outputs = []
    for _ in range(2):
        model = make_model(4)
        model.fit(X, y, CrossEntropyLoss(), Adam(model.parameters()), epochs=5, random_state=4, verbose=False)
        outputs.append(model.predict(X))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_fit_prints_progress(capsys, small_spirals):
    X, y = small_spirals
    model = make_model(4)
    model.fit(X, y, CrossEntropyLoss(), SGD(model.parameters()), epochs=3, random_state=4)
    assert "epoch 3/3" in capsys.readouterr().out


def test_predict_in_batches():
    model = make_model(5)
    X = np.random.default_rng(5).normal(size=(50, 2))
    np.testing.assert_allclose(model.predict(X, batch_size=7), model.forward(X))


def test_save_and_load(tmp_path):
    model = make_model(6)
    other = make_model(7)
    X = np.random.default_rng(8).normal(size=(10, 2))
    path = tmp_path / "weights.npz"
    model.save_weights(path)
    arrays_before = [p.data for p in other.parameters()]
    other.load_weights(path)
    np.testing.assert_allclose(other.predict(X), model.predict(X))
    for p, before in zip(other.parameters(), arrays_before):
        assert p.data is before


def test_load_wrong_architecture(tmp_path):
    path = tmp_path / "weights.npz"
    make_model(9).save_weights(path)
    other = Sequential([Dense(2, 8), ReLU(), Dense(8, 3)])
    with pytest.raises(ValueError):
        other.load_weights(path)


def test_summary():
    assert "total parameters: 195" in make_model(10).summary()
