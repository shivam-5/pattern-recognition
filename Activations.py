import numpy as np

class Activations:
    def __init__(self):
        pass

    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        z = self.sigmoid(x)
        return z * (1 - z)

    # Tanh
    def tanh(self, x):
        return 2 * self.sigmoid(2 * x) - 1

    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2

    # Log sigmoid
    def log_sigmoid(self, x):
        return np.log(self.sigmoid(x))

    # Relu
    def relu(self, x):
        return np.where(x >= 0, x, 0)

    def relu_derivative(self, x):
        return np.where(x >= 0, 1, 0)

    # Leaky Relu
    def leaky_relu(self, x):
        return np.where(x >= 0, x, 0.01 * x)

    def leaky_relu_derivative(self, x):
        return np.where(x >= 0, 1, 0.01)

    # Parameterized Relu
    def param_relu(self, x, a):
        return np.where(x >= 0, x, a * x)

    def param_relu_derivative(self, x, a):
        return np.where(x >= 0, 1, a)

    # Exponential relu
    def exp_relu(self, x, a):
        return x if x >= 0 else a * (np.exp(x) - 1)

    def exp_relu_derivative(self, x, a):
        return 1 if x >= 0 else self.exp_relu(x) + a

    # Swish
    def swish(self, x):
        return x * self.sigmoid(x)

    # Softmax
    def softmax(self, x):
        z = np.exp(x)
        return z / np.sum(z)

    def softmax_derivative(self, x):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        z = x.reshape(-1, 1)
        return np.diagflat(z) - np.dot(z, z.T)

    # Heaviside
    def heaviside(self, x):
        return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))
