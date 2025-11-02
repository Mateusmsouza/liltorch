import pytest
import numpy as np
from liltorch.nn.loss import MeanSquaredError


class TestMeanSquaredError:
    def setup_method(self):
        self.mse = MeanSquaredError()

    def test_forward_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        result = self.mse.forward(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(result, expected), "Forward MSE computation failed."

    def test_forward_zero_error(self):
        y_true = np.array([5.0, -3.2, 7.1])
        y_pred = np.array([5.0, -3.2, 7.1])
        result = self.mse.forward(y_true, y_pred)
        assert np.isclose(
            result, 0.0), "MSE should be zero when predictions match."

    def test_forward_shape_mismatch(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
            self.mse.forward(y_true, y_pred)

    def test_backward_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        grad = self.mse.backward(y_true, y_pred)
        expected = 2 * (y_pred - y_true) / y_true.size
        np.testing.assert_allclose(
            grad, expected, err_msg="Gradient computation failed.")

    def test_backward_zero_error(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        grad = self.mse.backward(y_true, y_pred)
        expected = np.zeros_like(y_true)
        np.testing.assert_allclose(
            grad, expected, err_msg="Gradient should be zero when predictions match.")

    def test_backward_negative_values(self):
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.5, -2.5, -3.5])
        grad = self.mse.backward(y_true, y_pred)
        expected = 2 * (y_pred - y_true) / y_true.size
        np.testing.assert_allclose(
            grad, expected, err_msg="Backward pass failed with negative values.")
