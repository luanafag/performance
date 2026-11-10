import numpy as np
from copy import deepcopy
from rbg.Utilities import cv_score

class GridSearch:
    def __init__(self, model, param_grid, metrics, cv, greater_is_better = False):
        """Exhaustive search over specified parameter values for an estimator.
        
        Tests all possible mathematical combinations from a hyperparameter dictionary
        to find the best configuration. After finding the best combination 
        through cross-validation, the winning estimator is refitted using the entire dataset.

        Args:
            model (class or object): The model to be optimized. It must have 'fit' and 'predict' methods.
            param_grid (dict): dictionary with parameter names (str) as keys and lists of parameter settings
            to try as values.
            metrics (callable): a scoring function that takes true targets and predictions and returns a single
            float score.
            cv (class or callable): Cross-Validation method used to split the data. KFold, LOOCV or Holdout.
            greater_is_better (bool, optional): control parameter for the evaluation metric. Defaults to False.
            For metrics such as Accuracy, Recall, R², precision, F-1 score, it must be True.
            For metrics such as MSE, RMSE, MAE, it must be False.
        """
        self.model = model
        self.param_grid = param_grid 
        self.cv = cv
        self.metrics = metrics
        self.greater_is_better = greater_is_better
        self.best_params_ = None
        self.best_model_ = None
        self.best_score_ = None

    def _gen_combs(self):
        combinations = [{}]
        for key, values in self.param_grid.items():
            new_combs = []
            for comb in combinations:
                for val in values:
                    new_comb = comb.copy()
                    new_comb[key] = val
                    new_combs.append(new_comb)
            combinations = new_combs
        return combinations

    def fit(self, X, y = None, groups = None):
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        model_obj = self.model() if isinstance(self.model, type) else self.model
        if self.greater_is_better:
            best_score = float('-inf')
            for params in self._gen_combs():
                current_model = deepcopy(model_obj)
                current_model.set_params(**params) 
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        else:
            best_score = float('inf')
            for params in self._gen_combs():
                current_model = deepcopy(model_obj)
                current_model.set_params(**params) 
                score = cv_score(current_model, X, y, self.cv, self.metrics, groups=groups)
                if score < best_score:
                    best_score = score
                    self.best_params_ = params.copy()
        if self.best_params_ is None:
            raise RuntimeError("GridSearch failed to find a valid score for any parameter combination.")
        self.best_model_ = deepcopy(model_obj)
        self.best_model_.set_params(**self.best_params_)
        if y is not None:
            self.best_model_.fit(X, y)
        else:
            self.best_model_.fit(X)
        self.best_score_ = best_score
        return self

    def set_params(self, hyperparameters):
        for key, value in hyperparameters.items():
            self.param_grid[key] = value
        return self

    def get_params(self):
        return self.param_grid

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("GridSearch hasn't been fitted yet")
        return self.best_model_.predict(X)

