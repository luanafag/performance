"""Regression models, with one module per estimator.

Import models from their modules, as in rbg.Classification:
    from rbg.Regression.Linear import Linear
    from rbg.Regression.Ridge import Ridge
    from rbg.Regression.Lasso import Lasso
    from rbg.Regression.Polynomial import Polynomial
    from rbg.Regression.GLM import GLM
"""


from rbg.Regression.linearmodel import LinearModel
from .baseestimator import BaseEstimator
from rbg.Regression.GLM import GLM
from rbg.Regression.Lasso import Lasso
from rbg.Regression.Linear import Linear
from rbg.Regression.Polynomial import Polynomial
from rbg.Regression.Ridge import Ridge
