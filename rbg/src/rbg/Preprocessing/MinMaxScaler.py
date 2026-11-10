import numpy as np
from ._base import BaseTransformer, check_array 

class MinMaxScaler(BaseTransformer):
    """ Scale each feature to a given range, (0, 1) by default.

    Args:
        feature_range: Tuple '(a, b)' with the desired output range.add()
        Atributes:
            data_min_: Pert-feature minimum seen in the data during fit.
            data_max_: Per-feature maximum seen in the data during fit.
            data_range_: Per-feature range, 'data_max_ - data_min_', with zeros replaced by 1 to avoid division by zero.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        
    def fit(self, X: np.ndarray, y = None):
        """Compute the per-feature minimun and maximum.add()
        

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y: Ignored. Present for Application compatibility. Defaults to None. 

        Returns:
            The fitted scaler 'self'.
        """
        X = check_array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        
        data_range[data_range == 0] = 1.0 # cosntant column: avoid division by zero
        self.data_range_ = data_range
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale 'X' using the minimum and range learned in 'fit'.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
            the sacled data as new array.
        """
        
        self._check_fitted("data_min_", "data_range_")
        X = check_array(X)
        a, b = self.feature_range
        return a + (X - self.data_min_) / self.data_range_ * (b - a)

    def untransform(self,X):
        """untransform the data"""
        a,b = self.feature_range
        
        return (X - a)/(b-a) * (self.data_max_ - self.data_min_) + self.data_min_
     