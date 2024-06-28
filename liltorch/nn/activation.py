"""
This module implements commonly used activation functions for neural networks.

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns
in the data. This module provides functions for popular activations like ReLU, sigmoid, tanh, etc.
"""

import numpy as np

from liltorch.nn.layer import Layer


class Tanh(Layer):
    """
    TanH activation layer for neural networks.

    The Tanh (hyperbolic tangent) activation function introduces non-linearity into the network,
    allowing it to learn complex patterns. It maps input values between -1 and 1.

    This class implements the Tanh activation function for the forward and backward passes
    used during neural network training.
    """

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using the Tanh activation function.

        Args:
            input_data: A NumPy array representing the input data for this layer.

        Returns:
            A NumPy array containing the output of the Tanh activation function applied to the input data.
        """
        self.input = input_data
        return np.tanh(self.input)

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Calculates the gradients for the backward pass using the derivative of Tanh.

        Args:
            output_error: The error signal propagated from the subsequent layer during backpropagation.
                        (A NumPy array)
        learning_rate: The learning rate used for updating the weights during training. (float)

        Returns:
            A NumPy array containing the error signal to be propagated back to the previous layer.
        """
        return (1 - np.tanh(self.input) ** 2) * output_error
