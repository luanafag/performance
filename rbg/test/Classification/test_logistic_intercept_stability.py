"""Regression test: seven-class logistic fit must remain numerically stable.

From the repository root:
python -m pytest test/Classification/test_logistic_intercept_stability.py -v
Requires only NumPy, pytest and rbg. No external dataset or network.
"""

import warnings

import numpy as np
import pytest

from rbg.Classification.Logistic import Logistic


@pytest.fixture
def dados_nao_separaveis():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 5))
    beta = rng.normal(size=(5, 6)) * 1.5
    logits = np.column_stack(
        (X @ beta + np.array([-4, -2, -1, 0, 1, 2]), np.zeros(600))
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilidades = np.exp(logits)
    probabilidades /= probabilidades.sum(axis=1, keepdims=True)
    y = (rng.random(600)[:, None] > probabilidades.cumsum(axis=1)).sum(axis=1)

    # Each anchor has ALL seven labels. The augmented anchors span R^6,
    # excluding both complete and quasi-complete linear separation.
    ancoras = np.vstack((np.zeros((1, 5)), np.eye(5), -np.eye(5)))
    X = np.vstack((X, np.repeat(ancoras, 7, axis=0)))
    y = np.concatenate((y, np.tile(np.arange(7), len(ancoras))))
    assert np.linalg.matrix_rank(np.column_stack((np.ones(len(X)), X))) == 6
    return X, y


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_logistic_converge_sem_separacao(dados_nao_separaveis, fit_intercept):
    X, y = dados_nao_separaveis
    modelo = Logistic(
        classes=np.arange(7), reference_class=6, fit_intercept=fit_intercept
    )
    modelo.set_params(tol=1e-6, max_iter=100)

    with warnings.catch_warnings(), np.errstate(
        over="raise", invalid="raise", divide="raise"
    ):
        warnings.simplefilter("error", UserWarning)
        modelo.fit(X, y)
        probabilidades = modelo.predict_proba(X)

    assert probabilidades.shape == (len(X), 7)
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1, atol=1e-10)

    # A finite probability vector alone does not prove a converged fit.
    design = np.column_stack((np.ones(len(X)), X)) if fit_intercept else X
    respostas = (y[:, None] == np.arange(6)[None, :]).astype(float)
    score = design.T @ (respostas - probabilidades[:, :6])
    assert np.linalg.norm(score) / len(X) < 1e-6
