import numpy as np
import pytest

from rbg.Preprocessing import ZScoreClipper

# mean = 2, std = 1
X = np.array([[1.0], [3.0]])


def test_fit_learns_bounds():
    clipper = ZScoreClipper().fit(X)
    assert np.allclose(clipper.lower_, [-1.0])  # 2 - 3 * 1
    assert np.allclose(clipper.upper_, [5.0])   # 2 + 3 * 1


def test_custom_cutoff():
    clipper = ZScoreClipper(courtoff=1.0).fit(X)
    assert np.allclose(clipper.lower_, [1.0])
    assert np.allclose(clipper.upper_, [3.0])


def test_transform_clips_outliers():
    clipper = ZScoreClipper().fit(X)
    result = clipper.transform([[-10.0], [2.0], [10.0]])
    assert np.allclose(result, [[-1.0], [2.0], [5.0]])


def test_values_inside_bounds_are_unchanged():
    result = ZScoreClipper().fit_transform(X)
    assert np.allclose(result, X)


def test_constant_column_is_not_clipped():
    X_const = np.array([[5.0], [5.0], [5.0]])
    result = ZScoreClipper().fit_transform(X_const)
    assert np.allclose(result, X_const)


def test_each_column_has_its_own_bounds():
    X2 = np.array([[1.0, 10.0], [3.0, 30.0]])  # means 2 and 20, stds 1 and 10
    clipper = ZScoreClipper(courtoff=1.0).fit(X2)
    result = clipper.transform([[100.0, 100.0]])
    assert np.allclose(result, [[3.0, 30.0]])


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ZScoreClipper().transform(X)