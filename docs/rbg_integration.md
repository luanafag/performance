
# RBG Library Integration

## Overview

The RBG (Regressão Bem Geral) library is the main machine learning
library used in this project.

It is maintained in a separate repository.

The purpose of this repository is to apply the implementations
available in RBG to real-world datasets.

## Classification Models

The following classifiers are available in the current RBG version:

- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Logistic Regression

Example imports:

```python
from rbg.Classification import KNN, LDA, QDA
```

## Regression Models

The current RBG version includes:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Generalized Linear Models

Example imports:

```python
from rbg.Regression.Linear import Linear
from rbg.Regression.Ridge import Ridge
from rbg.Regression.Lasso import Lasso
from rbg.Regression.Polynomial import Polynomial
```

## Preprocessing

The RBG library provides a MinMaxScaler implementation.

Example import:

```python
from rbg.Preprocessing import MinMaxScaler
```

The scaler must be fitted only on training data.

## Model Selection

The library includes K-Fold Cross-Validation and hyperparameter
selection methods.

Example imports:

```python
from rbg.ModelSelection import KFold, GridSearch
```

## Known Integration Issues

<!-- TODO: Document the corrections required in the RBG library,
including the Utilities import issue, train_test_split implementation,
and compatibility between cv_score and StratifiedKFold. -->

## Installation

<!-- TODO: Add the official GitHub repository URL and the exact
installation command once the RBG repository is publicly available. -->

## Additional Models

<!-- TODO: Document the neural network implementation and any other
models added to the library. -->
