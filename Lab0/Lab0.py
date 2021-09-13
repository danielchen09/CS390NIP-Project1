import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
from collections.abc import Iterator

# Setting random seeds to keep everything deterministic.
from typing import Tuple

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"


ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


# Activation function.
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# Activation prime function.
def sigmoid_(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


# Batch generator for mini-batches. Not randomized.
def generate_batches(data_set: np.ndarray, step: int) -> Iterator[np.ndarray]:
    for i in range(0, len(data_set), step):
        yield data_set[i: i + step]


class Model:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs=100000, use_mini_batch=True, mini_batch_size=100):
        return self

    def predict(self, x: np.ndarray):
        pass


class NeuralNetwork2(Model):
    def __init__(self, input_size: int, output_size: int, neurons_per_layer: int, lr=0.1):
        super(NeuralNetwork2, self).__init__(input_size, output_size)
        self.neurons_per_layer = neurons_per_layer
        self.lr = lr
        self.W1 = np.random.randn(self.input_size, self.neurons_per_layer)
        self.W2 = np.random.randn(self.neurons_per_layer, self.output_size)
        self.b1 = random.random() * 2 - 1
        self.b2 = random.random() * 2 - 1
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.pred = None

    # Training with backpropagation.
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs=3000, use_minibatch=False, mbs=200) -> Model:
        y_original = y_train
        y_train = to_categorical(y_train, NUM_CLASSES)

        if not use_minibatch:
            mbs = x_train.shape[0]

        m = x_train.shape[0]
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            if use_minibatch:
                sample_indices = np.random.choice(m, mbs)
                x_batch = x_train[sample_indices, :]
                y_batch = y_train[sample_indices, :]
            else:
                x_batch = x_train
                y_batch = y_train
            self.forward(x_batch)
            dw1, dw2, db1, db2 = self.backward(x_batch, y_batch)
            self.W1 -= self.lr * dw1 / mbs
            self.W2 -= self.lr * dw2 / mbs
            self.b1 -= self.lr * db1 / mbs
            self.b2 -= self.lr * db2 / mbs
            eval_results((x_train, y_original), self.forward(x_train))
        return self

    # Forward pass.
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.pred = sigmoid(self.z2)
        return self.pred

    def backward(self, x: np.ndarray, y: np.ndarray):
        delta2 = self.loss_derivative(self.pred, y) * sigmoid_(self.z2)
        dw2 = self.a1.T @ delta2
        delta1 = (delta2 @ self.W2.T) * sigmoid_(self.z1)
        dw1 = x.T @ delta1
        return [dw1, dw2, sum(delta1), sum(delta2)]

    def loss_derivative(self, pred, y):
        return -y / pred + (1 - y) / (1 - pred)

    # Predict.
    def predict(self, x):
        return self.forward(x)


class Guesser(Model):
    def __init__(self, input_size: int, output_size: int):
        super(Guesser, self).__init__(input_size, output_size)

    def predict(self, x: np.ndarray):
        rows: int = x.shape[0]
        guess: np.ndarray = np.zeros(rows, self.output_size[0])
        for i in range(rows):
            guess[i][random.randint(0, self.output_size[0])] = 1
        return guess


class TfModel(Model):
    def __init__(self, input_size: int, output_size: int):
        super(TfModel, self).__init__(input_size, output_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(10)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['accuracy'])

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs=5, use_mini_batch=True, mini_batch_size=100) -> Model:
        self.model.fit(x_train, y_train, epochs=5, verbose=1)
        return self

    def predict(self, x: np.ndarray):
        return self.model.predict(x)


# =========================<Pipeline Functions>==================================

def get_raw_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(x_train.shape))
    print("Shape of yTrain dataset: %s." % str(y_train.shape))
    print("Shape of xTest dataset: %s." % str(x_test.shape))
    print("Shape of yTest dataset: %s." % str(y_test.shape))
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(raw_data):
    ((x_train, y_train), (x_test, y_test)) = raw_data
    x_train = x_train.reshape(-1, IMAGE_SIZE)
    x_test = x_test.reshape(-1, IMAGE_SIZE)
    x_train = x_train / 255
    x_test = x_test / 255
    print("New shape of x_train dataset: %s." % str(x_train.shape))
    print("New shape of x_test dataset: %s." % str(x_test.shape))
    print("New shape of y_train dataset: %s." % str(y_train.shape))
    print("New shape of y_test dataset: %s." % str(y_test.shape))
    return (x_train, y_train), (x_test, y_test)


def train_model(data) -> Model:
    x_train, y_train = data
    input_size = x_train.shape[1]

    if ALGORITHM == "guesser":
        return Guesser(input_size, NUM_CLASSES).train(x_train, y_train)
    elif ALGORITHM == "custom_net":
        return NeuralNetwork2(input_size, NUM_CLASSES, 128).train(x_train, y_train)
    elif ALGORITHM == "tf_net":
        return TfModel(input_size, NUM_CLASSES).train(x_train, y_train)
    else:
        raise ValueError("Algorithm not recognized.")


def run_model(data, model: Model):
    if ALGORITHM not in ['guesser', 'custom_net', 'tf_net']:
        return ValueError("Algorithm not recognized.")
    return model.predict(data)


def eval_results(data, predictions):  # TODO: Add F1 score confusion matrix here.
    x_test, y_test = data
    acc = 0
    for i in range(predictions.shape[0]):
        if np.argmax(predictions[i]) == y_test[i]:
            acc = acc + 1
    accuracy = acc / predictions.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = get_raw_data()
    data = preprocess_data(raw)
    model = train_model(data[0])
    predictions = run_model(data[1][0], model)
    eval_results(data[1], predictions)


def test():
    raw = get_raw_data()
    train_data, test_data = preprocess_data(raw)
    model = train_model(train_data)
    predictions = run_model(test_data[0], model)
    eval_results(test_data, predictions)


if __name__ == '__main__':
    test()
