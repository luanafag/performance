"""Neural networks written with NumPy only.

Conventions used in the whole module:

- Data comes in batches: x has shape (N, D), one example per row.
- A layer keeps its weights as Parameter objects, each with .data and .grad.
- The 1/N of the batch mean only appears in the loss. Layers never divide by N.
- backward overwrites the gradients of a layer; it does not add to them.
- forward never changes the array it receives.
"""

from .activations import ReLU, Sigmoid, Tanh, softmax
from .data import batches
from .gradcheck import check_layer, numerical_gradient, relative_error
from .layers import Dense, Layer, Parameter
from .losses import CrossEntropyLoss, MSELoss
from .metrics import confusion_matrix
from .MLP import MLPClassifier, MLPRegressor
from .model import Sequential
from .optimizers import SGD, Adam
