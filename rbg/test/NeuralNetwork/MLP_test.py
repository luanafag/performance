import numpy as np
import pytest

from rbg.ModelSelection import GridSearch, KFold
from rbg.NeuralNetwork import MLPClassifier, MLPRegressor
from rbg.Utilities import MSE, Accuracy, cv_score


@pytest.fixture
def sine():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 1))
    return X, np.sin(3 * X[:, 0])


def test_classifier_learns_spirals(spirals):
    X, y = spirals
    model = MLPClassifier(hidden_layers=(32,), learning_rate=0.02, epochs=200, random_state=0)
    assert model.fit(X, y) is model
    assert Accuracy(model.predict(X), y) > 0.9
    assert len(model.history["loss"]) == 200


def test_classifier_keeps_original_labels(small_spirals):
    X, y = small_spirals
    names = np.array(["a", "b", "c"])
    model = MLPClassifier(epochs=5, random_state=0).fit(X, names[y])
    np.testing.assert_array_equal(model.labels, names)
    assert set(model.predict(X)) <= set(names)


def test_predict_proba(small_spirals):
    X, y = small_spirals
    proba = MLPClassifier(epochs=5, random_state=0).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_given_labels_with_missing_class(small_spirals):
    X, y = small_spirals
    keep = y != 2
    model = MLPClassifier(epochs=5, random_state=0, labels=[0, 1, 2]).fit(X[keep], y[keep])
    assert model.predict_proba(X).shape == (len(X), 3)


def test_unknown_label_raises(small_spirals):
    X, y = small_spirals
    with pytest.raises(ValueError):
        MLPClassifier(epochs=1, labels=[0, 1]).fit(X, y)


def test_one_class_raises(small_spirals):
    X, _ = small_spirals
    with pytest.raises(ValueError):
        MLPClassifier(epochs=1).fit(X, np.zeros(len(X)))


def test_regressor_learns_sine(sine):
    X, y = sine
    model = MLPRegressor(hidden_layers=(32,), activation="tanh", learning_rate=0.01, epochs=300, random_state=0)
    y_pred = model.fit(X, y).predict(X)
    assert y_pred.shape == y.shape
    assert MSE(y_pred, y) < 0.01


def test_regressor_two_outputs(sine):
    X, y = sine
    Y = np.column_stack([y, -y])
    assert MLPRegressor(epochs=2, random_state=0).fit(X, Y).predict(X).shape == (len(X), 2)


def test_regressor_accepts_1d_x(sine):
    X, y = sine
    assert MLPRegressor(epochs=2, random_state=0).fit(X[:, 0], y).predict(X[:, 0]).shape == y.shape


@pytest.mark.parametrize("model", [MLPClassifier(), MLPRegressor()])
def test_predict_before_fit_raises(model):
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))


def test_predict_wrong_number_of_features(sine):
    X, y = sine
    model = MLPRegressor(epochs=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 2)))


def test_same_random_state_same_model(sine):
    X, y = sine
    first = MLPRegressor(epochs=5, random_state=3).fit(X, y).predict(X)
    second = MLPRegressor(epochs=5, random_state=3).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)


def test_get_and_set_params():
    model = MLPRegressor()
    assert model.get_params()["hidden_layers"] == (64,)
    assert model.set_params(hidden_layers=[8, 8], learning_rate=0.1) is model
    assert model.get_params()["hidden_layers"] == (8, 8)
    assert model.get_params()["learning_rate"] == 0.1


def test_set_unknown_param_raises():
    with pytest.raises(ValueError):
        MLPClassifier().set_params(alpha=1.0)


def test_set_params_discards_fit(sine):
    X, y = sine
    model = MLPRegressor(epochs=1, random_state=0).fit(X, y)
    model.set_params(epochs=2)
    with pytest.raises(RuntimeError):
        model.predict(X)


@pytest.mark.parametrize("params", [{"activation": "softplus"}, {"optimizer": "rmsprop"}])
def test_bad_activation_or_optimizer(sine, params):
    X, y = sine
    with pytest.raises(ValueError):
        MLPRegressor(epochs=1, **params).fit(X, y)


def test_cv_score(small_spirals):
    X, y = small_spirals
    score = cv_score(MLPClassifier(epochs=5, random_state=0), X, y, KFold(k=3, shuffle=True), Accuracy)
    assert 0.0 <= score <= 1.0


def test_grid_search_picks_more_epochs(spirals):
    X, y = spirals
    model = MLPClassifier(hidden_layers=(32,), learning_rate=0.02, random_state=0)
    search = GridSearch(model, {"epochs": [1, 100]}, Accuracy, KFold(k=3, shuffle=True), greater_is_better=True)
    search.fit(X, y)
    assert search.best_params_ == {"epochs": 100}
    assert search.predict(X).shape == y.shape
