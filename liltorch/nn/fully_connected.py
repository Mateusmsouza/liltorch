import random

import numpy as np

from liltorch.nn.layer import Layer
from liltorch.nn.loss import mse


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        """ apply input * weigths + bias"""
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= lr * weights_error
        self.bias -= lr * output_error
        return input_error
