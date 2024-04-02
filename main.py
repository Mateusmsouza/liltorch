from neuralnet.fully_connected import FullyConnectedLayer

if __name__ == "__main__":
    x_train = [[[0,0]], [[0,1]], [[1,0]], [[1,1]]]
    fc = FullyConnectedLayer(2, 3)
    print(fc.forward(x_train[0]))
