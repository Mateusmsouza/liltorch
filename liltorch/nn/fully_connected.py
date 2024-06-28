import numpy as np

from liltorch.nn.layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        """apply input * weigths + bias"""
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, upstream_gradients, lr):
        # Calculate gradients to propagate to the previous layer (dL/dz[i]) given 
        # a previous layer gradient (dL/dz[i+1]) (which in forward pass would be next layer)
        downstream_gradients = np.dot(upstream_gradients, self.weights.T)
        
        # Calculate local gradients for weights and biases (dL/dW and dL/dB )
        local_gradients_w = np.dot(self.input.T, upstream_gradients)
        local_gradients_b = np.sum(upstream_gradients, axis=0, keepdims=True)
        
        # Update weights and biases using the gradients and learning rate
        self.weights -= lr * local_gradients_w
        self.bias -= lr * local_gradients_b
        
        return downstream_gradients
