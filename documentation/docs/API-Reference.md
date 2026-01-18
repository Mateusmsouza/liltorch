<a id="__init__"></a>

# \_\_init\_\_

<a id="nn"></a>

# nn

<a id="nn.loss"></a>

# nn.loss

This module implements commonly used loss functions for neural networks.

Loss functions measure the difference between the model's predictions and the ground truth labels.
Minimizing the loss function during training helps the model learn accurate representations.
This module provides functions for popular loss functions like mean squared error, cross-entropy, etc.

<a id="nn.loss.MeanSquaredError"></a>

## MeanSquaredError Objects

```python
class MeanSquaredError()
```

Class to compute the Mean Squared Error (MSE) and its gradient.

<a id="nn.loss.MeanSquaredError.forward"></a>

#### forward

```python
def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Compute the Mean Squared Error between true and predicted values.

**Arguments**:

- `y_true` _np.ndarray_ - True values.
- `y_pred` _np.ndarray_ - Predicted values.
  

**Returns**:

- `np.ndarray` - The mean squared error.
  

**Raises**:

- `ValueError` - If y_true and y_pred do not have the same shape.

<a id="nn.loss.MeanSquaredError.backward"></a>

#### backward

```python
def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Compute the gradient of the Mean Squared Error with respect to the predicted values.

**Arguments**:

- `y_true` _np.ndarray_ - True values.
- `y_pred` _np.ndarray_ - Predicted values.
  

**Returns**:

- `np.ndarray` - The gradient of the loss with respect to y_pred.

<a id="nn.layer"></a>

# nn.layer

<a id="nn.layer.Layer"></a>

## Layer Objects

```python
class Layer()
```

This is a abstract class

<a id="nn.network"></a>

# nn.network

<a id="nn.network.Network"></a>

## Network Objects

```python
class Network()
```

A basic neural network class for building and training multi-layer networks.

This class provides a framework for creating and using neural networks with customizable layers.
It supports adding different layer types (inherited from `liltorch.nn.layer.Layer`), performing
forward and backward passes for training, and updating layer weights using the provided learning rate.

<a id="nn.network.Network.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lr: float) -> None
```

Initializes a new neural network.

**Arguments**:

- `lr` - The learning rate used for updating the weights of the layers during training. (float)

<a id="nn.network.Network.add"></a>

#### add

```python
def add(layer: Layer) -> None
```

Adds a layer to the neural network.

This method allows you to build your network by sequentially adding different layer types
(e.g., `Tanh`, `Linear`, etc.) inherited from the `Layer` class.

**Arguments**:

- `layer` - An instance of a layer class from `liltorch.nn.layer`.

<a id="nn.network.Network.forward"></a>

#### forward

```python
def forward(x: np.ndarray) -> np.ndarray
```

Performs the forward pass through the network.

This method propagates the input data (`x`) through all the layers in the network,
applying their respective forward passes sequentially.

**Arguments**:

- `x` - The input data for the network, typically a NumPy array.
  

**Returns**:

  The output of the network after passing through all the layers. (NumPy array)

<a id="nn.network.Network.backward"></a>

#### backward

```python
def backward(error: np.ndarray)
```

Performs the backward pass for backpropagation.

This method calculates the gradients for all layers in the network using backpropagation.
It iterates through the layers in reverse order, starting from the output layer and
propagating the error signal back to the previous layers.

**Arguments**:

- `error` - The error signal from the loss function, typically a NumPy array.
  

**Returns**:

  The updated error signal to be propagated further back in the network during training
  (usually not used in the final output layer). (NumPy array)

<a id="nn.fully_connected"></a>

# nn.fully\_connected

<a id="nn.fully_connected.FullyConnectedLayer"></a>

## FullyConnectedLayer Objects

```python
class FullyConnectedLayer(Layer)
```

Fully-connected layer (dense layer) for neural networks.

This layer performs a linear transformation on the input data followed by a bias addition.
It's a fundamental building block for many neural network architectures.

During the forward pass, the input data is multiplied by the weight matrix and then added
to the bias vector. The resulting output is passed to the next layer in the network.

During the backward pass, the gradients are calculated for both the weights and biases
using backpropagation. These gradients are used to update the weights and biases
during training to improve the network's performance.

<a id="nn.fully_connected.FullyConnectedLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, output_size)
```

Initializes a fully-connected layer.

**Arguments**:

- `input_size` - The number of neurons in the previous layer (the size of the input vector). (int)
- `output_size` - The number of neurons in this layer (the size of the output vector). (int)

<a id="nn.fully_connected.FullyConnectedLayer.forward"></a>

#### forward

```python
def forward(input_data)
```

Performs the forward pass through the layer.

This method calculates the weighted sum of the input data and the bias vector.

**Arguments**:

- `input_data` - The input data for the layer, a NumPy array of shape (batch_size, input_size).
  

**Returns**:

  The output of the layer after applying the weights and bias, a NumPy array
  of shape (batch_size, output_size).

<a id="nn.fully_connected.FullyConnectedLayer.backward"></a>

#### backward

```python
def backward(upstream_gradients, lr)
```

Performs the backward pass for backpropagation in this layer.

This method calculates the gradients for the weights, biases, and the error signal
to be propagated back to the previous layer.

**Arguments**:

- `upstream_gradients` - The gradient signal from the subsequent layer in the network
  (a NumPy array of shape (batch_size, output_size)).
- `lr` - The learning rate used for updating the weights and biases during training. (float)
  

**Returns**:

  The gradient signal to be propagated back to the previous layer in the network
  (a NumPy array of shape (batch_size, input_size)).

<a id="nn.activation"></a>

# nn.activation

This module implements commonly used activation functions for neural networks.

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns
in the data. This module provides functions for popular activations like ReLU, sigmoid, tanh, etc.

<a id="nn.activation.Tanh"></a>

## Tanh Objects

```python
class Tanh(Layer)
```

TanH activation layer for neural networks.

The Tanh (hyperbolic tangent) activation function introduces non-linearity into the network,
allowing it to learn complex patterns. It maps input values between -1 and 1.

This class implements the Tanh activation function for the forward and backward passes
used during neural network training.

<a id="nn.activation.Tanh.forward"></a>

#### forward

```python
def forward(input_data: np.ndarray) -> np.ndarray
```

Performs the forward pass using the Tanh activation function.

**Arguments**:

- `input_data` - A NumPy array representing the input data for this layer.
  

**Returns**:

  A NumPy array containing the output of the Tanh activation function applied to the input data.

<a id="nn.activation.Tanh.backward"></a>

#### backward

```python
def backward(output_error: np.ndarray, learning_rate: float) -> np.ndarray
```

Calculates the gradients for the backward pass using the derivative of Tanh.

**Arguments**:

- `output_error` - The error signal propagated from the subsequent layer during backpropagation.
  (A NumPy array)
- `learning_rate` - The learning rate used for updating the weights during training. (float)
  

**Returns**:

  A NumPy array containing the error signal to be propagated back to the previous layer.

