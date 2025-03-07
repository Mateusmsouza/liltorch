import numpy as np

from liltorch.nn.layer import Layer


class Convolutional(Layer):

    def __init__(self, kernel_size: tuple[int, int], n_filers: int, stride: int =1 , padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filers
        self.stride = stride
        self.padding = padding
        self.weights = None

    def _initialize_weights(self, x_input):
        _, width, height, channels = x_input.shape
        shape = (width, height, channels, self.n_filters)
        print(shape)

    def forward(self, x_input: np.ndarray):
        self._initialize_weights(x_input)
