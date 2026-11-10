"""Public configuration and fitted coefficient contracts."""

import numpy as np
import pytest

from rbg.Regression.GLM import GLM
from rbg.Regression.Lasso import Lasso
from rbg.Regression.Linear import Linear
from rbg.Regression.Polynomial import Polynomial
from rbg.Regression.Ridge import Ridge


@pytest.mark.parametrize("estimator", [Linear, Ridge, Lasso])
def test_linear_estimators_expose_configuration_and_coefficients(estimator):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    model = estimator()
    assert model.get_params()["fit_intercept"] is True
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.coef

    model.set_params(fit_intercept=False)
    assert model.get_params()["fit_intercept"] is False
    assert model.fit(X, y).shape == (1,)
    np.testing.assert_array_equal(model.coef, model._beta)
    coefficients = model.coef
    coefficients[:] = 100
    assert not np.array_equal(model.coef, coefficients)

    model.set_params(fit_intercept=True)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.coef
    assert model.fit(X, y).shape == (2,)


@pytest.mark.parametrize("estimator", [Ridge, Lasso])
def test_penalty_set_params_is_used_by_fit(estimator):
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    model = estimator(fit_intercept=False)
    model.set_params(alpha=0.0)
    unpenalized = model.fit(X, y).copy()
    model.set_params(alpha=10.0)
    penalized = model.fit(X, y)
    assert model.get_params()["alpha"] == 10.0
    assert abs(penalized[0]) < abs(unpenalized[0])


def test_polynomial_configuration_rebuilds_estimator():
    model = Polynomial([1], method="ridge")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.arange(4.0)
    model.fit(X, y)
    assert model.coef.shape == (2,)
    model.set_params(degrees=[2], fit_intercept=False, alpha=2.0)
    assert model.get_params()["degrees"].tolist() == [2]
    assert model.get_params()["alpha"] == 2.0
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.coef
    assert model.fit(X, y).shape == (2,)


def test_glm_exposes_solver_configuration():
    family = object()
    model = GLM(family)
    model.set_params(tol=1e-8, max_iter=200)
    assert model.get_params() == {
        "fit_intercept": True,
        "inference": False,
        "family": family,
        "tol": 1e-8,
        "max_iter": 200,
    }
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.set_params(unknown=1)
