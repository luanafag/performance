import numpy as np 

class GroupKFold:
    """
    Group K-Fold cross-validation.

    Ensures that samples belonging to the same group
    are never separated between training and test sets.
    """

    def __init__(self, k: int = 5, shuffle: bool = False, random_state=42):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 1:
            raise ValueError("'k' must be an integer larger than 1.")
        self.k = int(k)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y = None, groups = None):
        """
        Generates train and test indices while keeping all samples from the same group together.

        Args:
            X (np.ndarray): dataset used to determine the number of samples.
            groups (np.ndarray): group identifier for each sample.

        Yields:
            Tuples of (train_idxs, test_idxs).
        """
        if groups is None:
            if y is not None:
                groups = y
                y = None
            else:
                raise ValueError("GroupKFold must have parameter 'groups'.")
        X = np.asarray(X)
        groups = np.asarray(groups)

        if X.shape[0] != groups.shape[0]:
            raise ValueError("X and groups must have the same number of samples.")
        if groups.ndim != 1:
            raise ValueError("'groups' must be a one-dimensional array.")

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.k > n_groups:
            raise ValueError(f"Number of splits ({self.k}) cannot be greater than the number of groups ({n_groups}).")

        rng = np.random.default_rng(self.random_state)
        group_sizes = np.bincount(group_indices)
        group_order = np.arange(n_groups)

        if self.shuffle:
            group_order = rng.permutation(group_order)

        group_order = group_order[np.argsort(-group_sizes[group_order], kind='stable')]
        fold_groups = [[] for _ in range(self.k)]
        fold_sizes = np.zeros(self.k, dtype=int)
        for group_idx in group_order:
            smallest_fold = np.argmin(fold_sizes)
            fold_groups[smallest_fold].append(group_idx)
            fold_sizes[smallest_fold] += group_sizes[group_idx]

        all_indices = np.arange(X.shape[0])
        for i in range(self.k):
            test_group_ids = fold_groups[i]
            test_mask = np.isin(group_indices, test_group_ids)

            test_idxs = all_indices[test_mask]
            train_idxs = all_indices[~test_mask]

            if self.shuffle:
                test_idxs = rng.permutation(test_idxs)
                train_idxs = rng.permutation(train_idxs)

            yield train_idxs, test_idxs