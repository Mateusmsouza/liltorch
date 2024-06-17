import numpy as np

from neuralnet.fully_connected import FullyConnectedLayer
from neuralnet.activation import ActivationLayerTanh
from neuralnet.loss import mse, mse_prime


if __name__ == "__main__":
    lr = 0.1
    epochs = 1000
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    layers = [
        FullyConnectedLayer(2, 3),
        ActivationLayerTanh(),
        FullyConnectedLayer(3, 1),
        ActivationLayerTanh(),
    ]

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(x_train)):
            sample = x_train[i]
            target = y_train[i]

            # forward
            # passing the sample through all neuralnet layers
            for layer in layers:
                sample = layer.forward(sample)

            epoch_loss += mse(target, sample)

            # backward pass
            error = mse_prime(target, sample)
            for layer in reversed(layers):
                error = layer.backward(error, lr)

        print(f'Epoch {epoch} -> Average loss {epoch_loss/len(x_train)}')

    # output = fc.forward(x_train[0])

    # print(f'output is {output} type {type(output)}')
    # loss = mse(y_true=y_test[0], y_pred=output[0])
    # print(loss)
    # error = fc.backward(loss, lr=lr)
    # print(output)
    # print(y_test[0])
