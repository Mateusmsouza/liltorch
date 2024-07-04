import numpy as np
from liltorch.nn.fully_connected import FullyConnectedLayer

def test_fc_initialization():
    fc = FullyConnectedLayer(input_size=2, output_size=3)
    assert fc.weights.shape == (2, 3)
    assert fc.bias.shape == (1, 3)

def test_fc_forward():
    """ fc layer forward is defined by y = x . wT + b """
    fc = FullyConnectedLayer(input_size=2, output_size=1)
    weights = fc.weights
    bias = fc.bias
    fake_data = np.random.rand(1, 2)
    expected = np.dot(fake_data, weights) + bias
    assert expected == fc.forward(fake_data)

def test_fc_backward():
    """ fc layer backward receives upstrem gradients,
    calculates local and downstream gradients then pass downstream"""
    lr = 0.001
    fc = FullyConnectedLayer(input_size=2, output_size=1)
    weights = fc.weights
    bias = fc.bias
    fc.input = np.random.rand(1, 2)
    fake_gradient = np.random.rand(1, 1)
    expected_downstream_gradients = np.dot(fake_gradient, weights.T)

    assert expected_downstream_gradients.all() == fc.backward(fake_gradient, lr).all()

