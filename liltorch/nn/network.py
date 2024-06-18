from liltorch.nn.layer import Layer

class Network:

    def __init__(self, lr: float) -> None:
        self.layers = []
        self.lr = lr

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, error):
        for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)