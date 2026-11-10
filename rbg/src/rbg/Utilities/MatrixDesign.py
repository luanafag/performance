import numpy as np

def MatrixDesign(X: np.ndarray, degree, intercept = True) -> np.ndarray:
    """Build the design matrix by expanding features to polynomial degrees and optionally adding an intercept.

    Args:
        X (np.ndarray): Data of shape (n_samples, n_features) containing the input features.
        degree (int, np.integer, list, or np.ndarray): The maximum polynomial degree for each feature. 
            If an integer, it is applied as a single element array.
        intercept (bool, optional): If True, a column of ones is prepended to the matrix. Defaults to True.

    Returns:
        np.ndarray: The design matrix of shape (n_samples, n_output_features) containing the intercept 
        (if requested) and the polynomial combinations of the features.
    
    Raises:
        ValueError: If the shapes or dimensions of 'X' and 'degree' are incompatible.
    """

    if isinstance(degree, (int, np.integer)):
        degree = np.array([degree])


    design =  design = np.empty((len(X), 0))

    if intercept:
        design = np.column_stack((design,np.ones(len(X))))

    
    for indx,d in enumerate(degree):       
        for p in range(1, d + 1):
            coluna_potencia = X[:, indx] ** p
            design = np.column_stack((design, coluna_potencia))

    return design