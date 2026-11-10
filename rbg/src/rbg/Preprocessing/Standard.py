import numpy as np

class Standard():
    """Standardizes features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self._mean = None
        self._std =None
        self._ep = 1e-11


    def fit(self,X):
        """
        Calculates and saves mean and std from training data to use later.
        """
        self._mean = np.mean(X,axis = 0)
        self._std = np.std(X,axis = 0)
        # return np.array((X - self._mean) / (self._std + self._ep))
        return self


    def transform(self,X):
        """
        Scales data using previously fitted mean and std,
        must be used after Standard.fit()
        """
        return np.array((X - self._mean) / (self._std + self._ep))

    def untransform(self,X):
        """
        Untransform the Data
        """

        return X * (self._std + self._ep) + self._mean
