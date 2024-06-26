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
batch_size = 16
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28*28)
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
    batch_begin = 0
    batch_end = batch_size - 1
    while batch_begin < dataset_size:
        data = x_train[batch_begin:batch_end]
        target = y_train[batch_begin:batch_end]

        # forward
        output = model.forward(data)
        epoch_loss += mse_loss(target, output)

        # backward pass
        error = mse_grad(target, output)
        model.backward(error)

        batch_begin += batch_size
        batch_end += batch_size
        batch_end = min(dataset_size, batch_end)

    print(f'Epoch {epoch} -> Average loss {epoch_loss/len(x_train)}')

# testing
correct = 0
total = 0
dataset_size = len(x_test)
batch_begin = 0
batch_end = batch_size - 1

while batch_begin < dataset_size -1:
    data = x_test[batch_begin:batch_end]
    target = y_test[batch_begin:batch_end]

    output = model.forward(data)

    correct += np.all(np.array(output) == np.array(target), axis=1).sum()
    #int(np.argmax(output) == np.argmax(target))
    total += 1

    batch_begin += batch_size
    batch_end += batch_size
    batch_end = min(dataset_size, batch_end)


print(f"correct {correct}")
print(f'Test Accuracy: {correct/total}')
