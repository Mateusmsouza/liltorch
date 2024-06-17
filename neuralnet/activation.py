import numpy as np

from neuralnet.layer import Layer


class ActivationLayerTanh(Layer):

    def forward(self, input_data):
        self.input = input_data
        return np.tanh(self.input)

    def backward(self, output_error, learning_rate):
        return (1 - np.tanh(self.input) ** 2) * output_error
