'''
for this file you may need to install Keras to easily load MNIST dataset.

pip install keras tensorflow
'''
from keras.datasets import mnist
from keras.utils import to_categorical

from liltorch.nn.fully_connected import FullyConnectedLayer
from liltorch.nn.activation import ActivationLayerTanh
from liltorch.nn.network import Network

from liltorch.nn.loss import mse_loss, mse_grad
import numpy as np

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

print(f'shape of xtrain {x_train.shape} xtrain {y_train.shape} and xtrain {x_test.shape} xtest {x_test.shape}')
# build model
model = Network(lr=lr)
model.add(FullyConnectedLayer(28*28, 100))
model.add(ActivationLayerTanh())
model.add(FullyConnectedLayer(100, 50))
model.add(ActivationLayerTanh())
model.add(FullyConnectedLayer(50, 10))
model.add(ActivationLayerTanh())

# training
for epoch in range(epochs):
    epoch_loss = 0
    dataset_size = len(x_train)
    for i in range(dataset_size):
        sample = x_train[i]
        target = np.expand_dims(y_train[i], axis=0)

        # forward
        output = model.forward(sample)
        epoch_loss += mse_loss(target, output)

        # backward pass
        error = mse_grad(target, output)
        model.backward(error)

    print(f'Epoch {epoch} -> Average loss {epoch_loss/len(x_train)}')

# testing
correct = 0
total = 0
for i in range(len(x_test)):
    sample = x_test[i]
    target = np.expand_dims(y_test[i], axis=0)

    output = model.forward(sample)

    correct += int(np.argmax(output) == np.argmax(target))
    total += 1
print(f'Test Accuracy: {correct/total}')