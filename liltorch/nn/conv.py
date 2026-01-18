# code inspired in https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_04_Computer_Vision/CNNs_from_scratch.ipynb#scrollTo=4a4HoUpxr29X
import numpy as np

from liltorch.nn.layer import Layer


class Conv2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = .1 * \
            np.random.randn(out_channels, in_channels,
                            kernel_size, kernel_size)
        self.bias = np.zeros((in_channels, 1))

    def foward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        batch_size, input_channels, height, width = input.shape
        self.padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding,
                                                            self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (height + 2 * self.padding -
                         self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding -
                        self.kernel_size) // self.stride + 1
        self.output = np.zeros(
            (batch_size, self.out_channels, output_height, output_width))

        # convolution operation
        for i in range(output_height):
            for j in range(output_width):
                region = self.padded_input[:, :, i*self.stride:i*self.stride +
                                           self.kernel_size, j*self.stride: j*self.stride+self.kernel_size]
                self.output[:, :, i, j] = np.tensordot(
                    region, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.bias.T

        return self.output


if __name__ == "__main__":
    x = np.arange(9).reshape(1, 1, 3, 3)
    print(x)

    convolution = Conv2d(in_channels=1, out_channels=1, kernel_size=2)
    kernel = np.ones(shape=(1, 1, 2, 2))
    convolution.w = kernel

    print(convolution.foward(x))
