import numpy as np

from liltorch.nn.layer import Layer


class Convolutional(Layer):

    def __init__(
        self, kernel_size: tuple[int, int], n_filers: int, padding: str, stride: int = 1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filers
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None

    def _initialize_weights(self, x_input):
        """Initialize weights"""

        _, width, height, channels = x_input.shape
        shape = (width, height, channels, self.n_filters)
        self.bias = self.bias = np.random.rand(1, 1, 1, channels) - 0.5
        self.weights = self.bias = np.random.rand(*shape) - 0.5

    def _config_padding(self, height, filter_shape_h, width, filter_shape_w):
        """config padding for conv layer
        if you wish to understand the formula applied check Andrej Karpathy's notes (https://cs231n.github.io/convolutional-networks/)
        specially the Spartial arrangement paragraph."""

        if self.padding == "same":
            pad_height = int(((height - 1) * self.stride + filter_shape_h - height) / 2)
            pad_width = int(((width - 1) * self.stride + filter_shape_w - width) / 2)

            n_H = height
            n_W = width

        elif self.padding == "valid":
            pad_height = 0
            pad_width = 0

            n_H = int((height - filter_shape_h) / self.stride) + 1
            n_W = int((width - filter_shape_w) / self.stride) + 1
        else:
            Exception(
                f"padding {self.padding} is an invalid option. Use valid or same."
            )
        return pad_height, n_H, pad_width, n_W

    def _apply_padding(self, input, n_col_padding):
        """
        Add padding to an input.

        :param input:[numpy array]: dataset of shape (m, height, width, depth)
        :param n_col_padding:[Tuple[int, int]]: number of columns to pad
        :return:[numpy array]: padded dataset
        """
        return np.pad(
            input,
            (
                (0, 0),
                (n_col_padding[0], n_col_padding[0]),
                (n_col_padding[1], n_col_padding[1]),
                (0, 0),
            ),
            "constant",
        )

    def forward(self, x_input: np.ndarray):
        if not self.weights and not self.bias:
            self._initialize_weights(x_input)
        num_samples, width, height, channels = x_input.shape
        filter_shape_h, filter_shape_w = self.kernel_size

        pad_height, n_H, pad_width, n_W = self._config_padding(
            height=height,
            filter_shape_h=filter_shape_h,
            width=width,
            filter_shape_w=filter_shape_w,
        )

        Z = np.zeros(shape=(num_samples, n_H, n_W, self.n_filters))
        X_pad = self._apply_padding(
            input=x_input, n_col_padding=(pad_height, pad_width)
        )

        for i in range(num_samples):
            x = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    # (x_min, y_min, x_max, y_max)
                    y_min = self.stride * h
                    y_max = y_min + filter_shape_h
                    x_min = self.stride * w
                    x_max = x_min + filter_shape_w


if __name__ == "__main__":
    cnn0 = Convolutional(kernel_size=(3, 3), n_filers=70, padding="same", stride=1)
    input = np.random.rand(128, 128, 3, 24)  # pseudo image batch of 24 images
    cnn0.forward(input)
    print(cnn0)
