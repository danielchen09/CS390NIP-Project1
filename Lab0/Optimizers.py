import numpy as np


class OptimizerParams:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.98):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2


class Optimizer:
    def __init__(self, weights, biases, lr=0.1):
        self.weights = weights
        self.biases = biases
        self.lr = lr

    def update(self, w_grads, b_grads, batch_size):
        pass

    def _update(self, w_grads, b_grads, batch_size):
        self.weights = [w - self.lr * dw / batch_size for w, dw in zip(self.weights, w_grads)]
        self.biases = [b - self.lr * db / batch_size for b, db in zip(self.biases, b_grads)]
        return self.weights, self.biases


class SGD(Optimizer):
    def update(self, w_grads, b_grads, batch_size):
        return self._update(w_grads, b_grads, batch_size)

    def __str__(self):
        return f'sgd(lr={self.lr})'


class Adam(Optimizer):
    def __init__(self, weights, biases, lr=0.1, beta1=0.9, beta2=0.98, epsilon=1e-8):
        super(Adam, self).__init__(weights, biases, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vws = [np.zeros_like(w) for w in weights]
        self.sws = [np.zeros_like(w) for w in weights]
        self.vbs = [np.zeros_like(b) for b in biases]
        self.sbs = [np.zeros_like(b) for b in biases]
        self.t = 0

    def update(self, w_grads, b_grads, batch_size):
        self.t += 1
        self.vws = [(self.beta1 * vw + (1 - self.beta1) * dw) / (1 - self.beta1 ** self.t)
                    for vw, dw in zip(self.vws, w_grads)]
        self.vbs = [(self.beta1 * vb + (1 - self.beta1) * db) / (1 - self.beta1 ** self.t)
                    for vb, db in zip(self.vbs, b_grads)]
        self.sws = [(self.beta2 * sw + (1 - self.beta2) * (dw ** 2)) / (1 - self.beta2 ** self.t)
                    for sw, dw in zip(self.sws, w_grads)]
        self.sbs = [(self.beta2 * sb + (1 - self.beta2) * (db ** 2)) / (1 - self.beta2 ** self.t)
                    for sb, db in zip(self.sbs, b_grads)]
        w_grads = [vw / (np.sqrt(sw) + self.epsilon) for vw, sw in zip(self.vws, self.sws)]
        b_grads = [vb / (np.sqrt(sb) + self.epsilon) for vb, sb in zip(self.vbs, self.sbs)]
        return self._update(w_grads, b_grads, batch_size)

    def __str__(self):
        return f'adam(b1={self.beta1},b2={self.beta2})'
