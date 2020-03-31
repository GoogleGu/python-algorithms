
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ml.models import LinearModel

from lib import datautil


def sk_usage():

    data = [(2.5, 21), (5.1, 47), (3.2, 27), (8.5, 75), (3.5, 30), (1.5, 20), (9.2, 88), (5.5, 60), (8.3, 81),
            (2.7, 25), (7.7, 85), (5.9, 62), (4.5, 41), (3.3, 42), (1.1, 17), (8.9, 95), (2.5, 30), (1.9, 24),
            (6.1, 67), (7.4, 69), (2.7, 30), (4.8, 54), (3.8, 35), (6.9, 76), (7.8, 86)]

    df = pd.DataFrame(data)
    X, Y = df.values[:, 0:1], df.values[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    plt.scatter(X, Y, marker='X')
    plt.plot(X_test, Y_pred)
    plt.show()



class VanillaLinearRegression(LinearModel):

    def loss(self, X, Y):
        return np.sum((X @ self.theta - Y) ** 2)

    def gradient(self, X, Y):
        return np.sum((X.T @ (X @ self.theta - Y)), axis=0)

    def predict(self, X):
        return self.pad_with_ones(X.copy()) @ self.theta


def demo():
    data_file = datautil.get_data_file_path('studentscores.csv')
    df = pd.read_csv(data_file)
    X, Y = df.values[:, 0:1], df.values[:, 1:]

    model = VanillaLinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    plt.scatter(X, Y, marker='X')
    plt.plot(X, Y_pred)
    plt.show()

    plt.plot(model.errors)
    plt.show()
