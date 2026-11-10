import pytest
import numpy as np
from rbg.ModelSelection import RandomizedSearch
from rbg.Utilities import RBGUniform
from rbg.ModelSelection import KFold
# tests to see if rbguniform works inside randomized search (i really hope so)
class DummyModel:
    def __init__(self, alpha=0.5, k=3):
        self.alpha = alpha
        self.k = k
    def fit(self, X, y):
        self.fitted_ = True
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    def get_params(self):
        return {"alpha": self.alpha, "k": self.k}
def dummy_metric(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
def test_randomized_search_rbg_uniform_sampling():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 2, size=60)
    param_dist = {
        'alpha': RBGUniform(0, 1),
        'k': [3, 5, 7]
    }
    search = RandomizedSearch(
        model=DummyModel(),
        param_dist=param_dist,
        metrics=dummy_metric,
        cv=KFold(k=3),
        n_iter=10,
        random_state=42,
        greater_is_better=True
    )
    search.fit(X, y)
    assert search.best_params_ is not None
    assert search.best_model_ is not None
    assert search.best_score_ is not None

    best_alpha = search.best_params_['alpha']
    assert isinstance(best_alpha, (float, np.floating))
    assert 0.0 <= best_alpha <= 1.0, f"Valor de alpha ({best_alpha}) fora do limite [0, 1]"
    preds = search.predict(X[:5])
    assert len(preds) == 5

def test_randomized_search_unfitted_predict_raises_error():
    param_dist = {'alpha': RBGUniform(0, 1)}
    search = RandomizedSearch(
        model=DummyModel(),
        param_dist=param_dist,
        metrics=dummy_metric,
        cv=KFold(k=3))
    with pytest.raises(RuntimeError):
        search.predict(np.ones((5, 4)))