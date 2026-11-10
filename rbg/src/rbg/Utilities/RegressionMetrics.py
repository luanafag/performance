"""Regression evaluation metrics.
Functions that compare predicted values against true target values and return a single scalar score. All functions expect two arrays of the same shape and operate element-wise.
"""
import numpy as np

def rss(y_predict: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the residual sum of squares (RSS).

    Args:
        y_predict (np.ndarray): Predicted values, shape (n,).
        y_true (np.ndarray): True target values, shape (n,).

    Returns:
        float: Sum of the squared differences between predicted and true values.
    """
    return np.sum((y_predict-y_true)**2)

def MSE(y_predict: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the mean squared error (MSE).

    Args:
        y_predict (np.ndarray): Predicted values, shape (n,).
        y_true (np.ndarray): True target values, shape (n,).

    Returns:
        float: Average of the squared differences between predicted and true values.
    """
    return np.mean((y_predict - y_true)**2)

def RMSE(y_predict: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the root mean squared error (RMSE).

    Args:
        y_predict (np.ndarray): Predicted values, shape (n,).
        y_true (np.ndarray): True target values, shape (n,).

    Returns:
        float: Square root of the mean squared error.
    """
    return np.sqrt(MSE(y_predict, y_true))
    
def MAE(y_predict: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the mean absolute error (MAE).

    Args:
        y_predict (np.ndarray): Predicted values, shape (n,).
        y_true (np.ndarray): True target values, shape (n,).

    Returns:
        float: Average of the absolute differences between predicted and true values.
    """
    return np.mean(np.abs(y_predict - y_true))

def RSquare(y_predict: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the coefficient of determination (R²).

    Args:
        y_predict (np.ndarray): Predicted values.
        y_true (np.ndarray): True target values.

    Returns:
        float: Fraction of the variance in y_true explained by y_predict.

    Raises:
        ValueError: If y_true is constant, since TSS is zero and R² is undefined.
    """
    tss = np.sum((y_true - np.mean(y_true))**2)
    if tss == 0:
        raise ValueError("R² undefined: y_true is constant.")

    return 1 - rss(y_predict, y_true) / tss
