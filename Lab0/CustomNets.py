from Utils import *
from Model import Model
import random
from Plotter import Plotter
from Optimizers import *


class NeuralNetwork(Model):
    def __init__(self, input_size: int, output_size: int, neurons_per_layer: list[int],
                 lr=0.1, optimizer='adam', activation='relu', optimizer_params=None, name=''):
        super(NeuralNetwork, self).__init__(input_size, output_size)
        self.name = name
        self.lr = lr

        self.weights = []
        self.biases = []
        neurons_per_layer = [input_size] + neurons_per_layer + [output_size]
        self.num_layers = len(neurons_per_layer)
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(neurons_per_layer[i], neurons_per_layer[i + 1]))
            self.biases.append(random.random() * 2 - 1)

        self.node_inputs = []
        self.node_outputs = []

        self.activation_name = activation
        if activation == 'relu':
            self.activation = relu
            self.activation_ = relu_
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_ = sigmoid_

        if optimizer_params is None:
            optimizer_params = OptimizerParams()
        if optimizer == 'sgd':
            self.optimizer = SGD(self.weights, self.biases, lr=optimizer_params.lr)
        elif optimizer == 'adam':
            self.optimizer = Adam(self.weights, self.biases,
                                  lr=optimizer_params.lr, beta1=optimizer_params.beta1, beta2=optimizer_params.beta2)

    # Training with backpropagation.
    def train(self, x_train, y_train, epochs=3000, use_minibatch=False, mbs=200) -> Model:
        y_original = y_train

        m = x_train.shape[0]
        if not use_minibatch:
            mbs = m

        plotter = Plotter(f'({self.name})custom_net_{self.activation_name}_{self.optimizer}_mbs{mbs}_eps{epochs}')
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
            grads = self.backward(y_batch)
            self.weights, self.biases = self.optimizer.update(*grads, mbs)
            accuracy = eval_results((0, y_original), self.forward(x_train), self.output_size)
            plotter.add_data(epoch, accuracy)
        plotter.save()
        return self

    # Forward pass.
    def forward(self, x):
        self.node_outputs = [x]
        self.node_inputs = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            self.node_inputs.append(self.node_outputs[-1] @ w + b)
            self.node_outputs.append(self.activation(self.node_inputs[-1]))
        self.node_inputs.append(self.node_outputs[-1] @ self.weights[-1] + self.biases[-1])
        self.node_outputs.append(sigmoid(self.node_inputs[-1]))
        return self.node_outputs[-1]

    def backward(self, y):
        w_grads = []
        b_grads = []
        delta = loss_(y, self.node_outputs[-1]) * sigmoid_(self.node_inputs[-1])
        w_grads.insert(0, self.node_outputs[-2].T @ delta)
        b_grads.insert(0, sum(delta))
        for i in range(1, self.num_layers - 1):
            delta = (delta @ self.weights[-i].T) * self.activation_(self.node_outputs[-i - 1])
            w_grads.insert(0, self.node_outputs[-i - 2].T @ delta)
            b_grads.insert(0, sum(delta))
        return w_grads, b_grads

    # Predict.
    def predict(self, x):
        return self.forward(x)


class NeuralNetwork2(Model):
    def __init__(self, input_size: int, output_size: int, neurons_per_layer: int, lr=0.1, name=''):
        super(NeuralNetwork2, self).__init__(input_size, output_size)
        self.name = name
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
    def train(self, x_train, y_train, epochs=3000, use_minibatch=False, mbs=200) -> Model:
        y_original = y_train

        if not use_minibatch:
            mbs = x_train.shape[0]
        m = x_train.shape[0]
        plotter = Plotter(f'({self.name})custom_net2_relu_{mbs}_eps{epochs}')
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
            accuracy = eval_results((0, y_original), self.forward(x_train), self.output_size)
            plotter.add_data(epoch, accuracy)
        plotter.save()
        return self

    # Forward pass.
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.pred = sigmoid(self.z2)
        return self.pred

    def backward(self, x, y):
        delta2 = self.loss_derivative(y) * sigmoid_(self.z2)
        dw2 = self.a1.T @ delta2
        delta1 = (delta2 @ self.W2.T) * relu_(self.z1)
        dw1 = x.T @ delta1
        return [dw1, dw2, sum(delta1), sum(delta2)]

    def loss_derivative(self, y):
        return -y / self.pred + (1 - y) / (1 - self.pred)

    # Predict.
    def predict(self, x):
        return self.forward(x)
