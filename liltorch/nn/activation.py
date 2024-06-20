import numpy as np

from liltorch.nn.layer import Layer


class ActivationLayerTanh(Layer):

    def forward(self, input_data):
        '''fordward pass using tanh activation'''
        self.input = input_data
        return np.tanh(self.input)

    def backward(self, output_error, learning_rate):
        '''backward pass using derivate of tanh'''
        return (1 - np.tanh(self.input) ** 2) * output_error
