from abc import abstractmethod

import numpy as np


THRESHOLD = 0.00001


class LinearModel:

    def __init__(self):
        self.theta = None
        self.errors = []

    def pad_with_ones(self, X):
        zeros = np.zeros((X.shape[0], 1))
        return np.hstack((zeros, X))

    @abstractmethod
    def loss(self, X, Y):
        pass

    def fit(self, X, Y, lr=0.1, iterations=1000):
        padded_X = self.pad_with_ones(X.copy())
        self.errors = []
        self.theta =np.random.rand(padded_X.shape[1], 1)
        m = padded_X.shape[0]
        last_error = 10000
        for i in range(iterations):
            step = (lr / m) * self.gradient(padded_X, Y)
            self.theta = self.theta - step
            self.errors.append(self.loss(padded_X, Y))
            this_error = self.errors[-1]
            if last_error - this_error < THRESHOLD:
                break
            last_error = this_error

    @abstractmethod
    def gradient(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
