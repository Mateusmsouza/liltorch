from neuralnet.fully_connected import FullyConnectedLayer

if __name__ == "__main__":
    data = [[1, 3, 4],
            [2, 6, 8]]
    fc = FullyConnectedLayer(3, 2)
    print(fc.forward(data))
