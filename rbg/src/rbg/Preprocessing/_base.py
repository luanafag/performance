import numpy as np

def check_array(X: np.asarray) -> np.asarray:
    """Check that the input is a 2D array.
    Args:
        X (np.asarray): Input array.
    Returns:
        np.asarray: 2D array.
    Raises:
        ValueError: If the input is not a 2D array.
    """
    
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Was expected a 2d array, but got an array with shape {X.ndim}D")
    
    return X

class BaseTransformer:
    """Base class for all transformers.
       Subclasses must implement 'fit' and 'transform' methods.
    """
    def fit(self, X, y=None):
        """Learn the transfrmation parameters from 'X'.

        Args:
            X: Training data of shape (n_samples, n_features). 
            y: Ignored. Present for Aplication compatibility. Defaults to None.
        returns:
            NotimplementedError: must be implemented in subclasses.
        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError  
    
    def transform(self, X):
        """Apply the learned transformation to 'X'.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            The transformed data.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError
    
    def fit_transform(self, X, y=None):
        """Fit to 'X', then transform it.add()

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for Application compatibility. Defaults to None.

        Returns:
            The transformed data.
        """
        return  self.fit(X, y).transform(X)
    
    def _check_fitted(self, *attrs):
        """Check that the transformer has been fit ted.
        
        Args:
        *attrs: Names of the attributes that 'fit' is expected to set.add()

        Raises:
            RuntimeError: If any of the attributes is missing.
        """
        for a in attrs:
            if not hasattr(self, a):
                raise RuntimeError(f"{type(self).__name__} was not fitted. Call 'fit' before using this method.")
