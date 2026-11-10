import numpy as np 
from ._base import BaseTransformer, check_array

class ZScoreClipper(BaseTransformer):
    """Clip each feature to 'mean +/- coutoff * std'.

    Args:
        courtoff: Number of standard deviations from the mean that are
            kept. Defaults to 3.0.
    Atributes:
        lower_: Per-feature lower bound.
        upper_: Per-feature upper bound.
    """

    def __init__(self, courtoff = 3.0) -> None:
        self.courtoff = courtoff
    
    def fit(self, X: np.ndarray, y = None) -> 'ZScoreClipper':
        """Compute the per-feature clipping bounds.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features). 
            y: Ignored. Present for Application compatibility. Defaults to None.

        Returns:
            The fitted clipper.
        """
        X = check_array(X)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0 # constant column: noting is clipped
        
        self.lower_ = mean - self.courtoff * std
        self.upper_ = mean + self.courtoff * std
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip 'X' to the bounds learned in 'fit'.add()

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
            The clipped data as a new array.
        Raises:
            RuntimeError: If called before 'fit'.
        """
        self._check_fitted("lower_", "upper_")
        X = check_array(X)
        return np.clip(X, self.lower_, self.upper_)