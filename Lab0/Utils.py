import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from Model import Model
import random

NUM_CLASSES = 10
IMAGE_SIZE = 784

# Activation function.
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# Activation prime function.
def sigmoid_(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return (x > 0) * 1


def loss(y_true, y_pred):
    return np.sum(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1), axis=0)


def loss_(y_true, y_pred):
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


def eval_metric(metric: tf.keras.metrics.Metric, y_true, y_pred):
    metric.update_state(to_categorical(y_true, NUM_CLASSES), to_categorical(y_pred, NUM_CLASSES))
    return metric.result().numpy()


def eval_accuracy(data, predictions):
    predictions = np.argmax(predictions, axis=1)
    acc = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == data[i]:
            acc = acc + 1
    return acc / predictions.shape[0]


def eval_results(data, predictions, long=False):
    x_test, y_test = data
    accuracy = eval_accuracy(y_test, predictions)
    predictions = np.argmax(predictions, axis=1)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    if long:
        print("Confusion Matrix:")
        print(tf.math.confusion_matrix(y_test, predictions, num_classes=NUM_CLASSES))
        recall = eval_metric(tf.keras.metrics.Recall(), y_test, predictions)
        precision = eval_metric(tf.keras.metrics.Precision(), y_test, predictions)
        print(f'F1 score: {2 * precision * recall / (precision + recall)}')


class Guesser(Model):
    def __init__(self, input_size: int, output_size: int):
        super(Guesser, self).__init__(input_size, output_size)

    def predict(self, x):
        rows= x.shape[0]
        guess = np.zeros(rows, self.output_size)
        for i in range(rows):
            guess[i][random.randint(0, self.output_size)] = 1
        return guess
