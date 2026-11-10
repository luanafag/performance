import numpy as np

class KFold:
    """K-Fold Cross Validation.
    
    Separates the dataset in k folds and uses k-1 folds as train and 1 fold as test.
    """
    def __init__(self, k : int = 5, shuffle : bool = False, random_state = 42):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 1:
            raise ValueError("The number of splits 'k' must be an integer greater than 1.")
        self.k = int(k)
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, data_size, y = None):
        if not isinstance(data_size, bool) and isinstance(data_size, (int, np.integer)):
            n_samples = int(data_size)
            if n_samples <= 0:
                raise ValueError("'data_size' must be positive.")
        elif hasattr(data_size, "__len__"):
            n_samples = len(data_size)
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if self.k > n_samples:
            raise ValueError(f"Number of splits 'k' cannot be greater than the number of samples.")
        if y is not None:
            y = np.asarray(y)
            if n_samples != y.shape[0]:
                raise ValueError("X and y dimensions must match.")
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        idx_sets = np.array_split(idxs, self.k)
        for i in range(self.k):
            test_idxs = idx_sets[i]
            train_idxs = np.concatenate([idx_sets[j] for j in range(self.k) if j!=i])
            yield train_idxs, test_idxs