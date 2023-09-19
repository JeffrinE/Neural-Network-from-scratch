import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
for _ in range(2):
  np.random.shuffle(data) # shuffling before splitting into testing and training sets

data_test = data[0:1000].T
y_dev = data_test[0]
x_dev = data_test[1:n]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_,m_train = x_train.shape

def init():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def ReLU(z):
    return np.maximum(z, 0)
def ReLU_dv(z):
    return z > 0
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_dv(z):
    return 1.0 / (1.0 + np.exp(-z))
def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    A1 = ReLU(z1)
    z2 = w2.dot(A1) + b2
    A2 = softmax(z2)
    return z1, A1, z2, A2


def backward_prop(z1, A1, z2, A2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = A2 - one_hot_y
    dw2 = 1 / m * dz2.dot(A1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU_dv(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2





def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def update(a, w1, b1, w2, b2, dw1, db1, dw2, db2):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    return w1, b1, w2, b2

def get_predic(A2):
    return np.argmax(A2, 0)

def get_accuracy(pred, y):
    print(pred, y)
    return np.sum(pred == y) / y.size

def gradient_descent(x, y, a, iter):
    w1, b1, w2, b2 = init()
    for i in range(iter):
        z1, A1, z2, A2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, A1, z2, A2, w1, w2, x, y)
        w1, b1, w2, b2 = update(a, w1, b1, w2, b2, dw1, db1, dw2, db2)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predic(A2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(x_train, y_train, 0.10, 500)


def make_predic(x, w1, b1, w2, b2):
    _, _, _, A2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predic(A2)
    return predictions


def test_prediction(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predic(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(3, w1, b1, w2, b2)
