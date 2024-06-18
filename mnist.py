'''
for this file you may need to install Keras to easily load MNIST dataset.

pip install keras tensorflow
'''
from keras.datasets import mnist
from keras.utils import to_categorical

from liltorch.nn.fully_connected import FullyConnectedLayer
from liltorch.nn.activation import ActivationLayerTanh
from liltorch.nn.network import Network

from liltorch.nn.loss import mse, mse_prime

# load MNIST from server
lr = 0.1
epochs = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# build model
model = Network(lr=lr)
model.add(FullyConnectedLayer(28*28, 100))
model.add(ActivationLayerTanh())
model.add(FullyConnectedLayer(100, 50))
model.add(ActivationLayerTanh())
model.add(FullyConnectedLayer(50, 10))
model.add(ActivationLayerTanh())

for epoch in range(epochs):
    epoch_loss = 0
    dataset_size = len(x_train)
    for i in range(dataset_size):
        sample = x_train[i]
        target = y_train[i]

        # forward
        output = model.forward(sample)
        print(f'output shape {output.shape} target shape {target.shape}')
        epoch_loss += mse(target, output)

        # backward pass
        error = mse_prime(target, output)
        model.backward(error)

    print(f'Epoch {epoch} -> Average loss {epoch_loss/len(x_train)}')
