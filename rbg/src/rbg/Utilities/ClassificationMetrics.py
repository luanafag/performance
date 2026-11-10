"""Classification evaluation metrics.
Functions that compare predicted class labels against true class labels.
All functions expect two arrays of the same shape. Accuracy returns a single scalar score; Precision, Recall and F1 return one score per class, ordered
as np.unique of the labels found in y_pred and y_true combined.
When a score is undefined (zero denominator), it is set to 'zero_division'.
"""

import numpy as np

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, zero_division: float = 0.0) -> np.ndarray:
    """Divide element-wise, replacing undefined results (zero denominator).

    Args:
        numerator (np.ndarray): Numerator values, shape (k,).
        denominator (np.ndarray): Denominator values, shape (k,).
        zero_division (float): Value returned where the denominator is zero.

    Returns:
        np.ndarray: numerator / denominator, with zero_division where the denominator is zero.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(numerator.shape, float(zero_division))
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result

def _class_counts(y_pred: np.ndarray, y_true: np.ndarray ) -> tuple:
    """Compute true positives, false positives and false negatives per class.

    Args:
        y_pred (np.ndarray): Predicted class labels, shape (n,).
        y_true (np.ndarray): True class labels, shape (n,).

    Returns:
        tuple: (tp, fp, fn), each an np.ndarray of shape (k,), where k is the
        number of distinct labels in y_pred and y_true combined.

    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    true_idx = np.searchsorted(classes, y_true)
    predict_idx = np.searchsorted(classes, y_pred)

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(confusion, (true_idx, predict_idx), 1)

    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    return tp, fp, fn

def Accuracy(y_pred: np.ndarray, y_true: np.ndarray ) -> float:
    """Compute the accuracy.

    Args:
        y_pred (np.ndarray): Predicted class labels, shape (n,).
        y_true (np.ndarray): True class labels, shape (n,).

    Returns:
        float: Fraction of predictions that match the true labels.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    if y_true.size == 0:
        raise ValueError("Accuracy undefined: empty arrays.")
    return float(np.mean(y_pred == y_true))

def Precision(y_pred: np.ndarray, y_true: np.ndarray, zero_division: float = 0.0) -> np.ndarray:
    """Compute the precision of each class.

    Args:
        y_pred (np.ndarray): Predicted class labels, shape (n,).
        y_true (np.ndarray): True class labels, shape (n,).
        zero_division (float): Value used for a class that was never predicted.

    Returns:
        np.ndarray: TP / (TP + FP) per class, shape (k,). Of everything predicted
        as a class, the fraction that really belonged to it.
    """
    tp, fp, _ = _class_counts(y_pred, y_true)
    return _safe_divide(tp, tp + fp, zero_division)

def Recall(y_pred: np.ndarray, y_true: np.ndarray, zero_division: float = 0.0) -> np.ndarray:
    """Compute the recall of each class.

    Args:
        y_pred (np.ndarray): Predicted class labels, shape (n,).
        y_true (np.ndarray): True class labels, shape (n,).
        zero_division (float): Value used for a class absent from y_true.

    Returns:
        np.ndarray: TP / (TP + FN) per class, shape (k,). Of everything that
        really belonged to a class, the fraction the model found.
    """
    tp, _, fn = _class_counts(y_pred, y_true)
    return _safe_divide(tp, tp + fn, zero_division)

def F1Score(y_pred: np.ndarray, y_true: np.ndarray, zero_division: float = 0.0) -> np.ndarray:
    """Compute the F1-score of each class.

    Args:
        y_pred (np.ndarray): Predicted class labels, shape (n,).
        y_true (np.ndarray): True class labels, shape (n,).
        zero_division (float): Value used for a class with no TP, FP or FN.

    Returns:
        np.ndarray: Harmonic mean of precision and recall per class, shape (k,),
        computed as 2·TP / (2·TP + FP + FN).
    """
    tp, fp, fn = _class_counts(y_pred, y_true)
    return _safe_divide(2 * tp, 2 * tp + fp + fn, zero_division)

def Gini(y_true:np.ndarray) :

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    ind_gin = sum(pmk*(1-pmk) for pmk in p)

    return ind_gin


def classif_error(y_true:np.ndarray):

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    return (1 -np.max(p))

    


def entropy(y_true:np.ndarray) :

    if y_true.size == 0:
        raise ValueError("The array is empty")
    _,ocurrencies = np.unique(y_true,return_counts= True)
    p = ocurrencies/len(y_true)

    entropy_result = sum(pmk*np.log(pmk) for pmk in p)


    return - entropy_result

 

def mode(y_true:np.ndarray, axis = None):

    y_true = np.asarray(y_true)
    if y_true.size == 0:
        raise ValueError("The array is empty")

    if axis is not None and y_true.ndim == 2:
        n_samples = y_true.shape[1]
        mode_results = []
        for i in range(n_samples):
            previsions_or_i = y_true[:,i]
            values, cont = np.unique(previsions_or_i, return_counts=True)
            ind = np.argmax(cont)
            mode_results.append(values[ind])
        return np.array(mode_results)

    else:
        values, cont = np.unique(y_true, return_counts = True)
        ind = np.argmax(cont)
        return values[ind]

def error_rate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return 1 - Accuracy(y_pred, y_true)
