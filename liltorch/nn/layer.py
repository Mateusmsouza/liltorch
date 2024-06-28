class Layer:
    """
    This is a abstract class
    """

    input = None
    output = None

    def foward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError
