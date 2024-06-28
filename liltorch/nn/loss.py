"""
This module implements commonly used loss functions for neural networks.

Loss functions measure the difference between the model's predictions and the ground truth labels.
Minimizing the loss function during training helps the model learn accurate representations.
This module provides functions for popular loss functions like mean squared error, cross-entropy, etc.
"""

import numpy as np


class MeanSquaredError:
    """
    Class to compute the Mean Squared Error (MSE) and its gradient.
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the Mean Squared Error between true and predicted values.

        Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

        Returns:
        np.ndarray: The mean squared error.

        Raises:
        ValueError: If y_true and y_pred do not have the same shape.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length.")
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Mean Squared Error with respect to the predicted values.

        Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

        Returns:
        np.ndarray: The gradient of the loss with respect to y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.size
