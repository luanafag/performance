import numpy as np
from copy import deepcopy
from rbg.Utilities import cv_score, RBGRandInt, RBGUniform

class RandomizedSearch:
    def __init__(self, model, param_dist, metrics, cv, n_iter: int = 10, random_state=42, greater_is_better=False):
        """Initializes a Randomized Search. param_dist must receive the parameters distributions as a dictionary, 
        and the hyperparameters are decided randomly. n_iter is the number of times the hyperparameters are decided,
        and the best set of hyperparameters is taken.
        
        In param_dist, you can use RBGUniform(a, b) [e.g. param_dist = {'learning_rate' : RBGUniform(0, 1)}]
        from our library RBG if you want a random float in the interval [a, b], or RBGRandInt(a, b) if 
        you want a random integer in the interval [a, b]. You can also use numpy functions such as
        np.linspace or np.logspace.
        """
        self.model = model
        self.param_dist = param_dist
        self.metrics = metrics
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.greater_is_better = greater_is_better

        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None

    def fit(self, X, y = None, groups = None):
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        model_obj = self.model() if isinstance(self.model, type) else self.model
        rng = np.random.default_rng(self.random_state)
        if self.greater_is_better:
            best_score = float('-inf')
            for _ in range(self.n_iter):
                params = {}
                for key, value in self.param_dist.items():
                    if callable(value):
                        params[key] = value(rng)
                    elif isinstance(value, (list,tuple,np.ndarray)):
                        val = rng.choice(value)
                        params[key] = val.item() if hasattr(val,"item") else val
                    else:
                        raise TypeError(f"The parameter '{key}' must be a list or a function (callable).")
                current_model = deepcopy(model_obj)
                current_model.set_params(**params)
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        else:
            best_score = float('inf')
            for _ in range(self.n_iter):
                params = {}
                for key, value in self.param_dist.items():
                    if callable(value):
                        params[key] = value(rng)
                    elif isinstance(value, (list,tuple,np.ndarray)):
                        val = rng.choice(value)
                        params[key] = val.item() if hasattr(val,"item") else val
                    else:
                        raise TypeError(f"The parameter '{key}' must be a list or a function (callable).")
                current_model = deepcopy(model_obj)
                current_model.set_params(**params)
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score < best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        self.best_model_ = deepcopy(model_obj)
        self.best_model_.set_params(**self.best_params_)
        if y is not None:
            self.best_model_.fit(X, y)
        else:
            self.best_model_.fit(X)
        self.best_score_ = best_score

        return self

    def set_params(self, param_dist = None, **kwargs):
        if param_dist is not None:
            for key, value in param_dist.items():
                self.param_dist[key] = value
        for key, value in kwargs.items():
            self.param_dist[key] = value
        return self

    def get_params(self):
        return self.param_dist

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("RandomizedSearch has not been fitted yet.")
        return self.best_model_.predict(X)