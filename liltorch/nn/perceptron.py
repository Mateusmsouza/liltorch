import numpy as np


class Perceptron:

    def __init__(self, x_dim: int, learning_rate:float = 0.1):
        self.lr = learning_rate
        self.weights = np.zeros(x_dim + 1)

    def forward(self, x):
        output = np.dot(x, self.weights[1:]) + self.weights[0]
        if output > 0:
            return True
        return False

    def backward(self, x, y, y_hat):
        self.weights[1:] = self.lr * (y-y_hat)*x
        self.weights[0] = self.lr * (y-y_hat)
