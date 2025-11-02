import pytest
import numpy as np
from liltorch.nn.activation import Tanh


class TestTanh:
    def setup_method(self):
        self.tanh = Tanh()

    def test_forward_basic(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = self.tanh.forward(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(
            result, expected, err_msg="Forward Tanh computation failed.")

    def test_forward_extreme_values(self):
        x = np.array([-100.0, 100.0])
        result = self.tanh.forward(x)
        # For large magnitudes, tanh(x) → -1 or +1
        expected = np.array([-1.0, 1.0])
        np.testing.assert_allclose(
            result, expected, atol=1e-7, err_msg="Tanh saturation for extreme values failed.")

    def test_forward_stores_input(self):
        x = np.array([0.5, -0.3])
        self.tanh.forward(x)
        assert np.allclose(
            self.tanh.input, x), "Forward pass should store input in self.input."

    def test_backward_basic(self):
        x = np.array([0.0, 1.0, -1.0])
        output_error = np.array([1.0, 0.5, -0.5])
        self.tanh.forward(x)  # must call forward first
        grad = self.tanh.backward(output_error, learning_rate=0.01)

        expected_grad = (1 - np.tanh(x) ** 2) * output_error
        np.testing.assert_allclose(
            grad, expected_grad, err_msg="Backward gradient computation failed.")

    def test_backward_large_values(self):
        x = np.array([100.0, -100.0])
        output_error = np.array([1.0, 1.0])
        self.tanh.forward(x)
        grad = self.tanh.backward(output_error, learning_rate=0.01)
        # derivative should approach 0 for saturated tanh
        expected_grad = (1 - np.tanh(x) ** 2) * output_error
        np.testing.assert_allclose(
            grad, expected_grad, atol=1e-9, err_msg="Gradient for large inputs incorrect.")
        assert np.all(
            np.abs(grad) < 1e-6), "Gradients for saturated tanh should be near zero."

    def test_backward_shape_preserved(self):
        x = np.random.randn(3, 4)
        output_error = np.random.randn(3, 4)
        self.tanh.forward(x)
        grad = self.tanh.backward(output_error, learning_rate=0.1)
        assert grad.shape == x.shape, "Backward pass should preserve shape."
