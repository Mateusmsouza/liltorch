import numpy as np

from liltorch.nn.fully_connected import FullyConnectedLayer
from liltorch.nn.activation import ActivationLayerTanh
from liltorch.nn.loss import mse, mse_prime
from liltorch.nn.network import Network


if __name__ == "__main__":
    lr = 0.1
    epochs = 1000
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    model = Network(lr=lr)
    model.add(FullyConnectedLayer(2, 3))
    model.add(ActivationLayerTanh())
    model.add(FullyConnectedLayer(3, 1))
    model.add(ActivationLayerTanh())

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(x_train)):
            sample = x_train[i]
            target = y_train[i]

            # forward
            output = model.forward(sample)
            epoch_loss += mse(target, output)

            # backward pass
            error = mse_prime(target, output)
            model.backward(error)

        print(f'Epoch {epoch} -> Average loss {epoch_loss/len(x_train)}')

    # output = fc.forward(x_train[0])

    # print(f'output is {output} type {type(output)}')
    # loss = mse(y_true=y_test[0], y_pred=output[0])
    # print(loss)
    # error = fc.backward(loss, lr=lr)
    # print(output)
    # print(y_test[0])
