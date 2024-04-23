import random

from linearmath.matrix import dot, add, transpose, subtract
from neuralnet.layer import Layer
from neuralnet.loss import mse


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        #TODO create a pandas like interface for dealing with scalars and tensors
        self.input_size = input_size
        self.weights = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [[random.uniform(-1, 1) for _ in range(output_size)]]

    def forward(self, input):
        """ apply input * weigths + bias"""
        self.input = input
        print(self.input)
        print(self.weights)
        self.output = add(dot(self.input, self.weights), self.bias)
        return self.output

    def backward(self, output_error, lr):
        input_error = dot(
            output_error,
            transpose(self.weights)
        )
        weights_error = dot(
            transpose(self.input),
            output_error
        )

        
        self.weights = self.weights - lr * weights_error
        self.bias = self.bias - lr * output_error
        return input_error