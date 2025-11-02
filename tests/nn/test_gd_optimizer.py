import pytest
import numpy as np
from liltorch.nn.optimizer import GD


class DummyLayer:
    """Mock layer com pesos, bias e gradientes simulados."""

    def __init__(self, has_weights=True):
        if has_weights:
            self.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
            self.bias = np.array([0.5, 0.5])
            self.local_gradients_w = np.array([[0.1, 0.2], [0.3, 0.4]])
            self.local_gradients_b = np.array([0.05, 0.05])
        else:
            # Layer sem pesos (ex: ReLU, Tanh, etc)
            self.weights = None
            self.bias = None
            self.local_gradients_w = None
            self.local_gradients_b = None


class DummyModel:
    """Mock network contendo layers e taxa de aprendizado."""

    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lr = lr


class TestGD:
    def setup_method(self):
        self.layer1 = DummyLayer()
        self.layer2 = DummyLayer(has_weights=False)
        self.layer3 = DummyLayer()
        self.model = DummyModel(
            layers=[self.layer1, self.layer2, self.layer3], lr=0.1)
        self.optimizer = GD(self.model)

    def test_step_updates_weights_and_bias(self):
        """Verifica se o método step atualiza pesos e bias corretamente."""
        old_w = self.layer1.weights.copy()
        old_b = self.layer1.bias.copy()

        self.optimizer.step()

        expected_w = old_w - self.model.lr * self.layer1.local_gradients_w
        expected_b = old_b - self.model.lr * self.layer1.local_gradients_b

        np.testing.assert_allclose(
            self.layer1.weights, expected_w, err_msg="Pesos não foram atualizados corretamente.")
        np.testing.assert_allclose(
            self.layer1.bias, expected_b, err_msg="Bias não foi atualizado corretamente.")

    def test_step_skips_layers_without_weights(self):
        """Garante que layers sem pesos sejam ignoradas no update."""
        self.optimizer.step()
        assert self.layer2.weights is None, "Layer sem pesos não deve ser alterada."
        assert self.layer2.bias is None, "Layer sem bias não deve ser alterada."

    def test_zero_grad_resets_gradients(self):
        """Verifica se zero_grad limpa todos os gradientes."""
        self.optimizer.zero_grad()
        for layer in self.model.layers:
            assert layer.local_gradients_w is None, "Gradiente de pesos não foi resetado."
            assert layer.local_gradients_b is None, "Gradiente de bias não foi resetado."
