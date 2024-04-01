import random

from linearmath.matrix import dot, add
from neuralnet.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        #TODO create a pandas like interface for dealing with scalars and tensors
        self.input_size = input_size
        self.weights = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [[random.uniform(-1, 1) for _ in range(output_size)]]

    def forward(self, input):
        """ apply input * weigths + bias"""
        self.input = input
        #TODO implement dot product into pandas like interface abstraction
        self.output = add(dot(self.weights, self.input), self.bias)
        return self.output

