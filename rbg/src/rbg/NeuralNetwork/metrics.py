"""Metrics that rbg.Utilities does not have yet.

For accuracy, precision, recall, F1 and the regression errors, use the
functions in rbg.Utilities.
"""

import numpy as np


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """Count hits and mistakes per class.

    Args:
        y_pred: Predicted classes, integers in [0, n_classes), shape (N,).
        y_true: True classes, same shape and range as y_pred.
        n_classes: Number of classes, so a class that never shows up still
            gets its row and column.

    Returns:
        Integer matrix of shape (n_classes, n_classes). M[i, j] counts the
        examples of class i that were predicted as class j.
    """
    M = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(M, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return M
