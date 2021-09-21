from CustomNets import *
from TfNets import *
from Dataset import *
import os
import numpy as np
import tensorflow as tf
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Information on dataset.

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
# ALGORITHM = "custom_net2"
# ALGORITHM = "tf_net"
ALGORITHM = "cnn"
# =========================<Pipeline Functions>==================================


def train_model(data, num_classes, epochs=100, use_minibatch=False, mbs=100,
                optimizer='sgd', optimizer_params=None, activation='relu', name='') -> Model:
    x_train, y_train = data
    input_size = x_train.shape[1]

    if ALGORITHM == "guesser":
        return Guesser(input_size, num_classes)\
            .train(x_train, y_train)
    elif ALGORITHM == "custom_net":
        return NeuralNetwork(input_size, num_classes, [128],
                             optimizer=optimizer, optimizer_params=optimizer_params, name=name, activation=activation)\
            .train(x_train, y_train, epochs=epochs, use_minibatch=use_minibatch, mbs=mbs)
    elif ALGORITHM == "custom_net2":
        lr = optimizer_params.lr if optimizer_params is not None else 0.1
        return NeuralNetwork2(input_size, num_classes, 128, lr=lr, name=name)\
            .train(x_train, y_train, epochs=epochs, use_minibatch=use_minibatch, mbs=mbs)
    elif ALGORITHM == "tf_net":
        return TfModel(input_size, num_classes)\
            .train(x_train, y_train, epochs=epochs)
    elif ALGORITHM == "cnn":
        return CNNModel(input_size, num_classes)\
            .train(x_train, y_train, epochs=epochs)
    else:
        raise ValueError("Algorithm not recognized.")


def run_model(data, model: Model):
    if ALGORITHM not in ['guesser', 'custom_net', 'custom_net2', 'tf_net', 'cnn']:
        return ValueError("Algorithm not recognized.")
    return model.predict(data)

# =========================<Main>================================================


def main():
    dataset = MNISTDataset(ALGORITHM == 'cnn')
    data = dataset.get_data()
    print(f'min and max: {np.min(data[0][0])}, {np.max(data[0][0])}')
    model = train_model(data[0], dataset.num_classes, name=dataset.name, epochs=20, use_minibatch=False, mbs=100)
    predictions = run_model(data[1][0], model)
    eval_results(data[1], predictions, dataset.num_classes, long=True)
    input('press enter to continue')


def run_test(dataset, epochs=100, use_minibatch=False, mbs=100,
                optimizer='sgd', optimizer_params=None, activation='relu', name=''):
    data = dataset.get_data()
    model = train_model(data[0], dataset.num_classes, epochs=epochs, use_minibatch=use_minibatch, mbs=mbs,
                        optimizer=optimizer, optimizer_params=optimizer_params, activation=activation, name=name)
    predictions = run_model(data[1][0], model)
    eval_results(data[1], predictions, True)


def run_mnist():
    global ALGORITHM
    ALGORITHM = 'custom_net'
    dataset = MNISTDataset(ALGORITHM == 'cnn')
    for mbs in [1, 10, 100, 500]:
        for i in range(5):
            run_test(dataset, epochs=1000, use_minibatch=True, mbs=mbs, name='mnist', optimizer='adam',
                     optimizer_params=OptimizerParams(lr=0.1, beta1=0.5 + 0.1 * i, beta2=0.5 + 0.1 * i + 0.099),
                     activation='relu')


if __name__ == '__main__':
    main()


