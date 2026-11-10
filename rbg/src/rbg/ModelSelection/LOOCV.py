from .KFold import KFold
import numpy as np

class LOOCV:
    """Leave-One-Out Cross Validation
    
    Take one sample and uses it as the testing set. Is a great option if the dataset is small enough.
    """
    def __init__(self):
        pass
    
    def split(self, data_size, y = None):
        if not isinstance(data_size, bool) and isinstance(data_size, (int, np.integer)):
            x = int(data_size)
        elif hasattr(data_size, "__len__"):
            x = len(data_size)
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if x < 2:
            raise ValueError("LOOCV requires at least 2 samples to perform cross-validation.")
        if y is not None:
            y = np.asarray(y)
            if x != y.shape[0]:
                raise ValueError("X and y dimensions must match.")
        return KFold(k=x).split(data_size = x, y=y)
