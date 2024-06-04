import random

import numpy as np

from linearmath.matrix import dot, add, transpose, subtract
from neuralnet.layer import Layer
from neuralnet.loss import mse


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        #TODO create a pandas like interface for dealing with scalars and tensors
        self.input_size = input_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def forward(self, input):
        """ apply input * weigths + bias"""
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update weights and bias
        print(f'weights_error is {weights_error} type {type(weights_error)}')
        print(f'output_error is {output_error} type {type(output_error)}')
        print(f'lr is {lr} type {type(lr)}')
        self.weights -= lr * weights_error
        self.bias -= lr * output_error
        return input_error