"""Solve the XOR problem with a small neural network trained via backpropagation.

This is a standalone, dependency-light (NumPy + matplotlib) version of the accompanying
Jupyter notebook. Running it trains a 2-2-1 network on the XOR truth table, prints the
results, and saves the training-loss and decision-boundary plots to ``assets/``.

Usage:
    python xor_backprop.py
"""

from __future__ import annotations

import os

import numpy as np

RNG_SEED = 42


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid, given its output ``a = sigmoid(z)``."""
    return a * (1.0 - a)


class XORNeuralNetwork:
    """A minimal 2-2-1 fully-connected network trained with backpropagation."""

    def __init__(self, n_input=2, n_hidden=2, n_output=1, learning_rate=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.learning_rate = learning_rate

        self.W1 = np.random.rand(n_input, n_hidden)
        self.b1 = np.random.rand(n_hidden)
        self.W2 = np.random.rand(n_hidden, n_output)
        self.b2 = np.random.rand(n_output)

        self.loss_history: list[float] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass and cache activations for backpropagation."""
        self.a1 = sigmoid(X @ self.W1 + self.b1)
        self.a2 = sigmoid(self.a1 @ self.W2 + self.b2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Backpropagate the error and update weights/biases in place."""
        delta_output = (y - output) * sigmoid_derivative(output)
        delta_hidden = (delta_output @ self.W2.T) * sigmoid_derivative(self.a1)

        self.W2 += self.a1.T @ delta_output * self.learning_rate
        self.b2 += np.sum(delta_output, axis=0) * self.learning_rate
        self.W1 += X.T @ delta_hidden * self.learning_rate
        self.b1 += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(self, X, y, epochs=10000, log_every=1000) -> list[float]:
        """Train for ``epochs`` iterations, recording the MSE loss each step."""
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            loss = float(np.mean((y - output) ** 2))
            self.loss_history.append(loss)
            self.backward(X, y, output)

            if log_every and (epoch == 1 or epoch % log_every == 0):
                print(f"Epoch {epoch:>6} / {epochs}  |  MSE loss: {loss:.6f}")
        return self.loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the network's raw output for the given inputs."""
        return self.forward(X)


def plot_results(model: XORNeuralNetwork, X: np.ndarray, y: np.ndarray, out_dir: str) -> None:
    """Save the training-loss curve and the learned decision boundary to ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend so the script runs without a display
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Training loss
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history, color="#2563eb", linewidth=2)
    plt.title("Training Loss (Mean Squared Error)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=110)
    plt.close()

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.25, 1.25, 300), np.linspace(-0.25, 1.25, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.85)
    plt.colorbar(contour, label="Network output")
    plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    for inp, target in zip(X, y.ravel().astype(int)):
        plt.scatter(*inp, c="white", edgecolors="black", s=220, zorder=3)
        plt.annotate(str(target), inp, ha="center", va="center", fontweight="bold")
    plt.title("Learned XOR Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.tight_layout()
    boundary_path = os.path.join(out_dir, "decision_boundary.png")
    plt.savefig(boundary_path, dpi=110)
    plt.close()

    print(f"\nSaved plots to '{loss_path}' and '{boundary_path}'.")


def main() -> None:
    np.random.seed(RNG_SEED)

    # XOR truth table
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = XORNeuralNetwork(learning_rate=0.5, seed=RNG_SEED)
    print("Training...\n")
    model.train(X, y, epochs=10000, log_every=1000)
    print("\nTraining complete.\n")

    predictions = model.predict(X)
    print(f"{'Input':<10}{'Target':<10}{'Raw output':<14}{'Predicted':<10}")
    print("-" * 44)
    for inp, target, pred in zip(X.astype(int), y.ravel().astype(int), predictions.ravel()):
        predicted_class = int(round(pred))
        mark = "OK" if predicted_class == target else "X"
        print(f"{str(inp):<10}{target:<10}{pred:<14.4f}{predicted_class:<10}{mark}")

    accuracy = float(np.mean(np.round(predictions) == y) * 100)
    print(f"\nAccuracy: {accuracy:.0f}%")

    plot_results(model, X, y, out_dir="assets")


if __name__ == "__main__":
    main()
