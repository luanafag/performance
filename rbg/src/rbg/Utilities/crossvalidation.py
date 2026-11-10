import numpy as np
import copy
# y_pred, y_true
def train_test_split(X : np.ndarray, y : np.ndarray, test_size : float = 0.2, shuffle = True, random_state = 42):
    """
    Divides the dataset between a training set and a testing set.

    Args:
        X (np.ndarray): input values.
        y (np.ndarray): output values.
        test_size (float, optional): proportion of samples assigned to the test set. Defaults to 0.2.
        shuffle (bool, optional): if True, the data will be shuffled. Defaults to True.
        random_state (int, optional): an integer seeds that ensures the random shuffle produces the same split every time. Defaults to 42.

    Raises:
        ValueError: if X and y dimensions don't match.
        ValueError: if the test_size is not a float between 0 and 1 neither a valid number of samples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if isinstance(test_size, int) and not isinstance(test_size,bool):
        if test_size <= 0 or test_size >= len(X):
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")
        n_test = test_size
    elif isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")
        n_test = int(np.ceil(test_size * len(X)))
    else:
        raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")

    n_samples = len(X)
    idxs = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        idxs = rng.permutation(idxs)
    test_idxs = idxs[:n_test]
    train_idxs = idxs[n_test:]
    X_train = X[train_idxs]
    X_test = X[test_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]
    return X_train, X_test, y_train, y_test

def train_val_test_split(X : np.ndarray, y : np.ndarray = None, val_size = 0.2, test_size = 0.2, shuffle = True, random_state = 42):
    """
    Splits arrays or matrices into random train, validation, and test subsets.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    if y is not None:
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
    if isinstance(test_size, int) and not isinstance(test_size,bool):
        if test_size <= 0 or test_size >= len(X):
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")
        n_test = test_size
    elif isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")
        n_test = int(np.ceil(test_size * len(X)))
    else:
        raise ValueError("'test_size' must represent the proportion or the exact number of samples on the test set.")
    if isinstance(val_size, float):
        n_val = int(np.ceil(n_samples * val_size))
    else:
        n_val = val_size
    if n_test + n_val >= n_samples:
        raise ValueError("The sum of test_size and val_size leaves no samples for training.")

    idxs = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        idxs = rng.permutation(idxs)
        
    test_idxs = idxs[:n_test]
    val_idxs = idxs[n_test : n_test + n_val]
    train_idxs = idxs[n_test + n_val:]
    
    X_train = X[train_idxs]
    X_val = X[val_idxs]
    X_test = X[test_idxs]
    if y is not None:
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        y_test = y[test_idxs]
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_val, X_test

def cv_score(model, X : np.ndarray, y : np.ndarray = None, cv_method = None, metrics = None, groups: np.ndarray = None):
    """Returns a list of scores given a machine learning model, a cross-validation method and a metric.
    
    Typical usage example:
        list_of_scores = cv_score(ex_model, X, y, KFold(10), "MSE")
        list_of_scores = cv_score(ex_model2, X, y, LOOCV, "MSE")

    Args:
        model: a machine learning model with methods .fit and .predict
        X (np.ndarray): input of the dataset.
        y (np.ndarray): output of the dataset.
        cv_method: a cross-validation method, KFold or LOOCV.
        metrics: a metric.

    Returns:
        float: the mean of the list of scores.
    """
    cv_obj = cv_method() if isinstance(cv_method, type) else cv_method
    scores = []
    if groups is not None:
        splits = cv_obj.split(X, y=y, groups=groups)
    else:
        splits = cv_obj.split(X, y)

    for idxs_train, idxs_test in splits:
        fold_model = copy.deepcopy(model)
        X_train = X[idxs_train]
        X_test = X[idxs_test]
        if y is not None:
            y_train = y[idxs_train]
            y_test = y[idxs_test]
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            score = metrics(y_pred, y_test)
        else:
            # aprendizado não supervisionado:
            fold_model.fit(X_train)
            X_pred = fold_model.predict(X_test)
            score = metrics(X_pred, X_test)
        scores.append(score)
    return np.mean(scores)