from liltorch.nn.layer import Layer


class Network:
    """
    A basic neural network class for building and training multi-layer networks.

    This class provides a framework for creating and using neural networks with customizable layers.
    It supports adding different layer types (inherited from `liltorch.nn.layer.Layer`), performing
    forward and backward passes for training, and updating layer weights using the provided learning rate.
    """

    def __init__(self, lr: float) -> None:
        """
        Initializes a new neural network.

        Args:
          lr: The learning rate used for updating the weights of the layers during training. (float)
        """
        self.layers = []
        self.lr = lr

    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the neural network.

        This method allows you to build your network by sequentially adding different layer types
        (e.g., `Tanh`, `Linear`, etc.) inherited from the `Layer` class.

        Args:
          layer: An instance of a layer class from `liltorch.nn.layer`.
        """
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the network.

        This method propagates the input data (`x`) through all the layers in the network,
        applying their respective forward passes sequentially.

        Args:
          x: The input data for the network, typically a NumPy array.

        Returns:
          The output of the network after passing through all the layers. (NumPy array)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, error: np.ndarray):
        """
        Performs the backward pass for backpropagation.

        This method calculates the gradients for all layers in the network using backpropagation.
        It iterates through the layers in reverse order, starting from the output layer and
        propagating the error signal back to the previous layers.

        Args:
          error: The error signal from the loss function, typically a NumPy array.

        Returns:
          The updated error signal to be propagated further back in the network during training
            (usually not used in the final output layer). (NumPy array)
        """
        for layer in reversed(self.layers):
            error = layer.backward(error, self.lr)
