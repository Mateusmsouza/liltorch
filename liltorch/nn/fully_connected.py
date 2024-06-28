import numpy as np

from liltorch.nn.layer import Layer


class FullyConnectedLayer(Layer):
    """
    Fully-connected layer (dense layer) for neural networks.

    This layer performs a linear transformation on the input data followed by a bias addition.
    It's a fundamental building block for many neural network architectures.

    During the forward pass, the input data is multiplied by the weight matrix and then added
    to the bias vector. The resulting output is passed to the next layer in the network.

    During the backward pass, the gradients are calculated for both the weights and biases
    using backpropagation. These gradients are used to update the weights and biases
    during training to improve the network's performance.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes a fully-connected layer.

        Args:
          input_size: The number of neurons in the previous layer (the size of the input vector). (int)
          output_size: The number of neurons in this layer (the size of the output vector). (int)
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        """
        Performs the forward pass through the layer.

        This method calculates the weighted sum of the input data and the bias vector.

        Args:
          input_data: The input data for the layer, a NumPy array of shape (batch_size, input_size).

        Returns:
          The output of the layer after applying the weights and bias, a NumPy array
          of shape (batch_size, output_size).
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, upstream_gradients, lr):
        """
        Performs the backward pass for backpropagation in this layer.

        This method calculates the gradients for the weights, biases, and the error signal
        to be propagated back to the previous layer.

        Args:
          upstream_gradients: The gradient signal from the subsequent layer in the network
                               (a NumPy array of shape (batch_size, output_size)).
          lr: The learning rate used for updating the weights and biases during training. (float)

        Returns:
          The gradient signal to be propagated back to the previous layer in the network
           (a NumPy array of shape (batch_size, input_size)).
        """
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
