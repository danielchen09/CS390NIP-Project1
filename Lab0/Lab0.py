from CustomNets import *
from TfNets import *
import os
import numpy as np
import tensorflow as tf
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "custom_net2"
# ALGORITHM = "tf_net"

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
        return NeuralNetwork(input_size, NUM_CLASSES, [128]).train(x_train, y_train, epochs=200, use_minibatch=True, mbs=50)
    elif ALGORITHM == "custom_net2":
        return NeuralNetwork2(input_size, NUM_CLASSES, 128).train(x_train, y_train, epochs=200)
    elif ALGORITHM == "tf_net":
        return TfModel(input_size, NUM_CLASSES).train(x_train, y_train)
    else:
        raise ValueError("Algorithm not recognized.")


def run_model(data, model: Model):
    if ALGORITHM not in ['guesser', 'custom_net', 'custom_net2', 'tf_net']:
        return ValueError("Algorithm not recognized.")
    return model.predict(data)

# =========================<Main>================================================


def main():
    raw = get_raw_data()
    data = preprocess_data(raw)
    model = train_model(data[0])
    predictions = run_model(data[1][0], model)
    eval_results(data[1], predictions, True)
    input()


if __name__ == '__main__':
    main()

