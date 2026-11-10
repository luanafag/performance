import numpy as np
import pytest

from rbg.Preprocessing import BaseTransformer, check_array


def test_check_array_keeps_2d():
    result = check_array([[1, 2], [3, 4]])
    assert result.shape == (2, 2)
    assert result.dtype == float


def test_check_array_reshapes_1d_to_column():
    result = check_array([1, 2, 3])
    assert result.shape == (3, 1)


def test_check_array_rejects_3d():
    with pytest.raises(ValueError):
        check_array(np.zeros((2, 2, 2)))


def test_base_fit_and_transform_not_implemented():
    base = BaseTransformer()
    with pytest.raises(NotImplementedError):
        base.fit([[1]])
    with pytest.raises(NotImplementedError):
        base.transform([[1]])


def test_check_fitted():
    base = BaseTransformer()
    with pytest.raises(RuntimeError):
        base._check_fitted("mean_")
    base.mean_ = 1.0
    base._check_fitted("mean_")  # should not raise