"""Sequential: a list of layers applied in order, and its training loop.

Sequential only uses the Layer interface (forward, backward, parameters), so
it works with any layer. For a model that plugs into rbg.ModelSelection, use
MLPClassifier or MLPRegressor.
"""

import numpy as np

from .data import batches
from .layers import Layer, Parameter
from .losses import CrossEntropyLoss, MSELoss
from .optimizers import SGD, Adam


class Sequential:
    """Network made of layers applied in order.

    Example:
        model = Sequential([Dense(2, 16), ReLU(), Dense(16, 3)])
    """

    def __init__(self, layers: list[Layer] | None = None) -> None:
        self.layers = list(layers) if layers is not None else []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run x, of shape (N, D), through every layer."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate dL/d(output); each layer stores its own gradients."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[Parameter]:
        """Return every Parameter, in layer order."""
        return [p for layer in self.layers for p in layer.parameters()]

    def named_parameters(self) -> list[tuple[str, Parameter]]:
        """Return (name, Parameter) pairs, with names like "0.W", "0.b", "2.W"."""
        return [(f"{i}.{p.name}", p) for i, layer in enumerate(self.layers) for p in layer.parameters()]

    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        loss: MSELoss | CrossEntropyLoss,
        optimizer: SGD | Adam,
    ) -> float:
        """Do one training step on one batch and return its loss."""
        value = loss.forward(self.forward(X_batch), y_batch)
        self.backward(loss.backward())
        optimizer.step()
        return value

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: MSELoss | CrossEntropyLoss,
        optimizer: SGD | Adam,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        random_state: int | np.random.Generator | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the network in place.

        Args:
            X: Training inputs of shape (N, D).
            y: Training targets, in the format the loss expects.
            loss: MSELoss or CrossEntropyLoss.
            optimizer: SGD or Adam, built with self.parameters().
            epochs: Number of passes over the data.
            batch_size: Examples per training step.
            validation_data: Optional (X_val, y_val), evaluated after each
                epoch and never trained on.
            random_state: Seed or generator for shuffling the batches.
            verbose: If True, print the progress about 10 times.

        Returns:
            {"loss": [...], "val_loss": [...]}, one value per epoch. "loss" is
            the mean training loss of the epoch; "val_loss" stays empty
            without validation data.
        """
        rng = np.random.default_rng(random_state)
        history = {"loss": [], "val_loss": []}
        print_every = max(1, epochs // 10)

        for epoch in range(epochs):
            batch_losses = []
            batch_sizes = []
            for X_batch, y_batch in batches(X, y, batch_size, random_state=rng):
                batch_losses.append(self.train_step(X_batch, y_batch, loss, optimizer))
                batch_sizes.append(len(X_batch))
            history["loss"].append(float(np.average(batch_losses, weights=batch_sizes)))

            if validation_data is not None:
                history["val_loss"].append(self.evaluate(*validation_data, loss))

            if verbose and ((epoch + 1) % print_every == 0 or epoch + 1 == epochs):
                text = f"epoch {epoch + 1}/{epochs}  loss {history['loss'][-1]:.4f}"
                if validation_data is not None:
                    text += f"  val_loss {history['val_loss'][-1]:.4f}"
                print(text)

        return history

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Return the network output for X (logits, for classification).

        Runs in batches so that big datasets do not use too much memory.
        """
        outputs = [self.forward(X[start:start + batch_size]) for start in range(0, len(X), batch_size)]
        return np.concatenate(outputs, axis=0)

    def classify(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Return the index of the predicted class of each example, shape (N,)."""
        return self.predict(X, batch_size).argmax(axis=1)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: MSELoss | CrossEntropyLoss,
        batch_size: int = 1024,
    ) -> float:
        """Return the mean loss over (X, y) without training."""
        total = 0.0
        for start in range(0, len(X), batch_size):
            X_batch = X[start:start + batch_size]
            y_batch = y[start:start + batch_size]
            total += loss.forward(self.forward(X_batch), y_batch) * len(X_batch)
        return total / len(X)

    def save_weights(self, path: str) -> None:
        """Write the parameters to a .npz file, keyed by name ("0.W", ...)."""
        np.savez(path, **{name: p.data for name, p in self.named_parameters()})

    def load_weights(self, path: str) -> None:
        """Copy the weights of a file written by save_weights into this model.

        Raises:
            ValueError: If the file does not match the architecture.
        """
        with np.load(path) as f:
            saved = set(f.files)
            for name, p in self.named_parameters():
                if name not in saved:
                    raise ValueError(f"load_weights: '{name}' is not in {path}.")
                if f[name].shape != p.data.shape:
                    raise ValueError(f"load_weights: '{name}' has shape {f[name].shape} in the file but {p.data.shape} in the model.")
                p.data[...] = f[name]
                saved.remove(name)
            if saved:
                raise ValueError(f"load_weights: the file has extra weights {sorted(saved)}.")

    def summary(self) -> str:
        """Return one line per layer with its number of parameters, and the total."""
        lines = []
        total = 0
        for i, layer in enumerate(self.layers):
            count = sum(p.data.size for p in layer.parameters())
            total += count
            lines.append(f"{i:>3}  {repr(layer):<30} {count:>10}")
        lines.append(f"total parameters: {total}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Sequential([{', '.join(repr(layer) for layer in self.layers)}])"
