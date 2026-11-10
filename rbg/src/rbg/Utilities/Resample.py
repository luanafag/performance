import numpy as np

def Resample(X : np.ndarray, size = None, n_resamples : int = 100, random_state = None, replace = True):
    """
    Returns resamples of the original dataset.
    If size = None and replace = True (by default), is a bootstrap.

    Args:
        X (np.ndarray): the original dataset
        size (optional): the size of each resample. Defaults to None.
        n_resamples (int, optional): the number of resamples. Defaults to 100.
        random_state (optional): seed for shuffling. Defaults to None.
        replace (bool, optional): if True, samples can be repeated. Defaults to True.

    Raises:
        ValueError: if 'n_resamples' is not an integer greater than 0.
        ValueError: if 'size' is not an integer greater than 0.
        ValueError: if replace = False and 'size' is greater than the size of X.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("'n_resamples' must be an integer greater than 0.")
    if size == None:
        size = n_samples
    else:
        if type(size) is not int or size<=0:
            raise ValueError("'size' must be an integer greater than 0.")
        if type(size) is int and replace == False and size > n_samples:
            raise ValueError("'size' must be at most the size of the dataset.")
    rng = np.random.default_rng(random_state)
    all_idxs = np.arange(n_samples)
    resamples = []
    for _ in range(n_resamples):
        train_idxs = rng.choice(all_idxs, size=size, replace=replace)
        test_idxs = np.setdiff1d(all_idxs, train_idxs)
        resamples.append((train_idxs, test_idxs)) # acho q o bootstrap devia retornar os testes tbm, se ele for usado como um metodo de cv
    return resamples # isso gasta mais memoria do que o yield, mas os testes pedem pra retornar uma lista ou array
