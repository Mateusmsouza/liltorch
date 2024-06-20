import numpy as np


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    return np.mean(np.power(y_true-y_pred, 2));

def mse_grad(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;