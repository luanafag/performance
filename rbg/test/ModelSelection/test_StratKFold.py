import numpy as np
import pytest

from rbg.Utilities import cv_score
from rbg.ModelSelection import KFold, StratifiedKFold


class DummyModel:
    """Modelo mock para testes isolados."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.ones(len(X))


def dummy_metric(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


@pytest.fixture
def sample_data():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


def test_kfold_split_accepts_y(sample_data):
    """Garante que KFold.split() aceita 'y' como 2º argumento posicional sem lançar TypeError."""
    X, y = sample_data
    kf = KFold(k=5)
    
    # Não deve lançar TypeError ao passar y
    try:
        splits = list(kf.split(X, y))
    except TypeError as e:
        pytest.fail(f"KFold.split(X, y) falhou com TypeError: {e}")
        
    assert len(splits) == 5


def test_kfold_split_without_y(sample_data):
    """Garante que KFold.split() funciona passando apenas X."""
    X, _ = sample_data
    kf = KFold(k=5)
    
    splits = list(kf.split(X))
    assert len(splits) == 5


def test_cv_score_with_kfold_and_y(sample_data):
    """Testa a execução de cv_score usando KFold com rótulos y."""
    X, y = sample_data
    model = DummyModel()
    
    score = cv_score(model, X, y, cv_method=KFold(k=5), metrics=dummy_metric)
    assert isinstance(score, (float, np.floating))


def test_cv_score_with_kfold_without_y(sample_data):
    """Testa cv_score para cenários não supervisionados (sem informar y)."""
    X, _ = sample_data
    
    class UnsupervisedDummyModel:
        def fit(self, X):
            return self
        def predict(self, X):
            return np.zeros(len(X))

    def unsupervised_metric(y_pred, X_test):
        return float(np.mean(y_pred))

    score = cv_score(
        UnsupervisedDummyModel(), 
        X, 
        y=None, 
        cv_method=KFold(k=5), 
        metrics=unsupervised_metric
    )
    assert score == 0.0


def test_cv_score_with_stratified_kfold(sample_data):
    """Testa cv_score integrado ao StratifiedKFold."""
    X, y = sample_data
    model = DummyModel()
    
    score = cv_score(model, X, y, cv_method=StratifiedKFold(k=5), metrics=dummy_metric)
    assert isinstance(score, (float, np.floating))


def test_stratified_kfold_raises_error_without_y(sample_data):
    """Garante que StratifiedKFold gera erro amigável se y for omitido ou None."""
    X, _ = sample_data
    skf = StratifiedKFold(k=5)
    
    with pytest.raises((ValueError, TypeError)):
        list(skf.split(X, y=None))