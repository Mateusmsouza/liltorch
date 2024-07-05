from liltorch.nn.network import Network

class GD:

    def __init__(self, model: Network) -> None:
        self.model = model

    def step(self):
        '''Update weights and biases using the gradients and learning rate'''
        for layer in reversed(self.model.layers):
            if layer.weights is not None:
                layer.weights -= self.model.lr * layer.local_gradients_w
                layer.bias -= self.model.lr * layer.local_gradients_b

    def zero_grad(self):
        ''' Reset gradients '''
        for layer in reversed(self.model.layers):
            layer.local_gradients_w = None
            layer.local_gradients_b = None
