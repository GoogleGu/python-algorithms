from abc import abstractmethod

import numpy as np


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

    def fit(self, X, Y, a=0.01, iterations=10000):
        padded_X = self.pad_with_ones(X.copy())
        self.errors = []
        self.theta =np.random.rand(*padded_X.shape)
        m = padded_X.shape[0]
        last_error = 100000
        for i in range(iterations):
            self.theta = self.theta - (a / m) * self.gradient(padded_X, Y)
            this_error = self.loss(padded_X, Y)
            self.errors.append(this_error)
            if last_error - this_error < 0.001:
                break
            last_error = this_error

    @abstractmethod
    def gradient(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
