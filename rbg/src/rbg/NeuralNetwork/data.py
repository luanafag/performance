"""Mini-batches for training.

For train/test splits use rbg.Utilities.train_test_split or
train_val_test_split.
"""

from collections.abc import Iterator

import numpy as np


def batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, y_batch) pairs that cover every example exactly once.

    Args:
        X: Inputs, one example per row.
        y: Targets, with the same number of rows as X.
        batch_size: Size of each batch. The last one can be smaller.
        shuffle: If False, the batches follow the original order.
        random_state: Seed or generator for the shuffle. Pass a generator to
            get a different order on every call.

    Raises:
        ValueError: If X and y have a different number of examples.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} examples but y has {len(y)}.")
    n = len(X)
    order = np.random.default_rng(random_state).permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]
