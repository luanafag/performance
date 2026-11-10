import numpy as np
import pytest

from rbg.Preprocessing import MinMaxScaler


X = np.array([[1.0, 10.0],
              [3.0, 20.0],
              [5.0, 30.0]])


def test_fit_learns_min_and_max():
    scaler = MinMaxScaler().fit(X)
    assert np.allclose(scaler.data_min_, [1.0, 10.0])
    assert np.allclose(scaler.data_max_, [5.0, 30.0])


def test_transform_default_range():
    result = MinMaxScaler().fit(X).transform(X)
    expected = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    assert np.allclose(result, expected)

def test_transform_custom_range():
    result = MinMaxScaler(feature_range=(-1, 1)).fit(X).transform(X)
    expected = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    assert np.allclose(result, expected)


def test_constant_column_does_not_divide_by_zero():
    X_const = np.array([[5.0], [5.0], [5.0]])
    result = MinMaxScaler().fit(X_const).transform(X_const)
    assert np.allclose(result, 0.0)


def test_transform_before_fit_raises():
    with pytest.raises(Exception):
        MinMaxScaler().transform(X)