from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import csv
import numpy as np
import random


def print_shapes(x_train, y_train, x_test, y_test):
    print("New shape of x_train dataset: %s." % str(x_train.shape))
    print("New shape of x_test dataset: %s." % str(x_test.shape))
    print("New shape of y_train dataset: %s." % str(y_train.shape))
    print("New shape of y_test dataset: %s." % str(y_test.shape))


class MNISTDataset:
    def __init__(self, is_cnn):
        self.image_size = 784
        self.num_classes = 10
        self.is_cnn = is_cnn
        self.name = 'mnist'

    def get_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        if self.is_cnn:
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
        else:
            x_train = x_train.reshape(-1, self.image_size)
            x_test = x_test.reshape(-1, self.image_size)

        x_train = x_train / 255
        x_test = x_test / 255
        print_shapes(x_train, y_train, x_test, y_test)
        return (x_train, y_train), (x_test, y_test)


class IrisDataset:
    def __init__(self, is_cnn):
        assert not is_cnn
        self.name = 'iris'
        with open('Iris/iris.data', newline='') as data:
            reader = csv.reader(data)
            x_raw = []
            y_raw = []
            for row in reader:
                if len(row) < 5:
                    continue
                x_raw.append(row[:4])
                y_raw.append(row[4])

        self.label, self.y = np.unique(y_raw, return_inverse=True)
        self.num_classes = len(self.label)
        self.y = self.y.reshape(-1, 1)
        self.x = np.array(x_raw, dtype=np.float32)
        self.size = self.x.shape[0]
        random_index = random.sample(range(self.size), self.size)
        self.x = self.x[random_index]
        self.y = self.y[random_index]

    def get_data(self):
        test_portion = 0.1
        x_train = self.x[:int(self.size * (1 - test_portion))]
        y_train = to_categorical(self.y[:int(self.size * (1 - test_portion))])
        x_test = self.x[:int(self.size * test_portion)]
        y_test = to_categorical(self.y[:int(self.size * test_portion)])

        print_shapes(x_train, y_train, x_test, y_test)
        return (x_train, y_train), (x_test, y_test)
