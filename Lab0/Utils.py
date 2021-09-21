import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from Model import Model
import random


# Activation function.
def sigmoid(x):
    x = np.maximum(np.minimum(x, 800), -800)
    return np.maximum(np.minimum(1 / (1 + np.exp(-x)), 0.9999), 0.0001)


# Activation prime function.
def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_(x):
    return (x > 0) * 1


def loss(y_true, y_pred):
    return np.sum(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1), axis=0)


def loss_(y_true, y_pred):
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


def eval_metric(metric: tf.keras.metrics.Metric, y_true, y_pred, num_classes):
    metric.update_state(to_categorical(y_true, num_classes), to_categorical(y_pred, num_classes))
    return metric.result().numpy()


def eval_accuracy(data, predictions):
    acc = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == data[i]:
            acc = acc + 1
    return acc / predictions.shape[0]


def confusion_matrix(data, predictions, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int)
    m = data.shape[0]
    for i in range(m):
        cm[predictions[i], data[i]] += 1
    return cm


def f1(cm, num_classes):
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm, axis=1)[i] - tp
        fn = np.sum(cm, axis=0)[i] - tp
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
    return 2 * recall * precision / (recall + precision)


def eval_results(data, predictions, num_classes, long=False):
    x_test, y_test = data
    predictions = np.argmax(predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = eval_accuracy(y_test, predictions)
    print("Classifier accuracy: %.2f%%" % (accuracy * 100))
    if long:
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, predictions, num_classes)
        print(cm)
        print('f1 scores:')
        f1s = f1(cm, num_classes)
        for i in range(num_classes):
            print(f'{i}: {f1s[i]:.4f}')
    return accuracy


class Guesser(Model):
    def __init__(self, input_size: int, output_size: int):
        super(Guesser, self).__init__(input_size, output_size)

    def predict(self, x):
        rows = x.shape[0]
        guess = np.zeros(rows, self.output_size)
        for i in range(rows):
            guess[i][random.randint(0, self.output_size)] = 1
        return guess
