import numpy as np

class Holdout:
    """Separates the dataset between one training set and one validation set, and can be used on cv_score.
    """
    def __init__(self, val_size=0.2, shuffle=False, random_state=42):
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data_size, y = None, groups = None):
        if isinstance(data_size, (int, np.integer)) and not isinstance(data_size, bool):
            n_samples = data_size
        elif hasattr(data_size, "__len__"):
            n_samples = len(data_size)
        elif isinstance(data_size, int) and not isinstance(data_size, bool):
            n_samples = data_size
        else:
            raise ValueError("'data_size' must be an integer or a dataset.")
        if n_samples <= 1:
            raise ValueError("'data_size' must contain at least 2 samples.")
        if isinstance(self.val_size, int) and not isinstance(self.val_size, bool):
            if self.val_size <= 0 or self.val_size >= n_samples:
                raise ValueError("'val_size' must be between 1 and n_samples-1.")
            n_val = self.val_size
        elif isinstance(self.val_size, float):
            if self.val_size <= 0 or self.val_size >= 1:
                raise ValueError("'val_size' must be between 0 and 1.")
            n_val = int(np.ceil(self.val_size * n_samples))
        else:
            raise ValueError("'val_size' must be an integer or a float.")
        n_train = n_samples - n_val
        idxs = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idxs = rng.permutation(idxs)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:]
        yield train_idxs, val_idxs