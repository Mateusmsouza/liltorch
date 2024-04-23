from neuralnet.fully_connected import FullyConnectedLayer
from neuralnet.loss import mse


if __name__ == "__main__":
    lr = 0.1
    # x_train = [
    #     [[0,0]], [[0,1]], [[1,0]], [[1,1]]
    #     ]
    # y_test = [
    #     [[0]], [[1]], [[1]], [[0]]
    # ]
    x_train = [
        [[1]], [[2]], [[4]], [[8]]
    ]
    y_test = [
        [2], [4], [8], [16]
    ]

    fc = FullyConnectedLayer(1, 1)

    output = fc.forward(x_train[0])
    loss = mse(y_true=y_test[0], y_pred=output[0])
    print(loss)
    error = fc.backward([[loss]], lr=lr)
    print(output)
    print(y_test[0])