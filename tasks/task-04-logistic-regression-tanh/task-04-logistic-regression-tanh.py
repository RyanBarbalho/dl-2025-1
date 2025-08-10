import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def tanh(self, z):
        ### START CODE HERE ###
        ### TODO: implement the tanh activation
        a = np.tanh(z)
        ### END CODE HERE ###
        return a

    def predict_proba(self, X):
        ### START CODE HERE ###
        ### TODO: compute activation output using tanh
        z = np.dot(X, self.weights) + self.bias
        a = self.tanh(z)
        ### END CODE HERE ###
        return a

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        a = self.tanh(z)
        # For tanh, threshold at 0: positive values -> class 1, negative values -> class 0
        prediction = np.where(a >= 0, 1, -1)
        return prediction

    def encode_labels(self, y):
        # Convert from {0, 1} to {-1, +1} for tanh activation
        encoded_y = np.where(y == 0, -1, 1)
        return encoded_y

    def train(self, X, y):
        ### START CODE HERE ###
        ### TODO: convert y from {0, 1} to {-1, +1}
        y_tanh = self.encode_labels(y)

        for _ in range(self.epochs):
            # Forward pass
            y_pred = self.predict_proba(X)

            # Compute error (difference between predicted and target)
            error = y_tanh - y_pred

            # Gradients
            # For tanh activation, derivative is (1 - tanh^2(z))
            # We need to compute this for each sample
            z = np.dot(X, self.weights) + self.bias
            tanh_derivative = 1 - np.tanh(z) ** 2

            # Gradient for weights: -2 * error * tanh_derivative * X
            grad_w = -2 * np.mean(
                error[:, np.newaxis] * tanh_derivative[:, np.newaxis] * X, axis=0
            )

            # Gradient for bias: -2 * mean(error * tanh_derivative)
            grad_b = -2 * np.mean(error * tanh_derivative)

            # Update parameters
            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b

            # Compute MSE loss
            loss = np.mean((y_tanh - y_pred) ** 2)
            self.loss_history.append(loss)
        ### END CODE HERE ###


def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a custom colormap for tanh outputs (-1 to +1)
    plt.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(label="Tanh Output (-1 to +1)")

    # Plot decision boundary at Z = 0
    plt.contour(
        xx, yy, Z, levels=[0], colors="black", linewidths=2, label="Decision Boundary"
    )

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Tanh Activation Decision Boundary")
    plt.legend()
    plt.show()


def plot_loss(model):
    plt.plot(model.loss_history, "k.")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MSE Loss over Training Iterations")
    plt.show()


def main():
    # Generate dataset
    X, y = generate_dataset()

    # Train the model
    neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
    neuron.train(X, y)

    # Plot decision boundary
    plot_decision_boundary(neuron, X, y)

    # Plot loss over training iterations
    plot_loss(neuron)


if __name__ == "__main__":
    main()
