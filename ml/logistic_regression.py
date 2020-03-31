import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib import datautil
from models import LinearModel


THRESHOLD = 0.0001

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class LogisticRegression(LinearModel):

    def loss(self, X, Y):
        prob = sigmoid(X @ self.theta)
        m = X.shape[0]
        return np.sum(-(Y.T @ np.log(prob) + (1-Y).T @ np.log(1 - prob)) / m)

    # def fit(self, X, Y):
    #     self.errors = []
    #     X = self.pad_with_ones(X)
    #     self.theta = np.random.normal(0, 1, (X.shape[1], 1))
    #
    #     for _ in range(self.iterations):
    #         step = self.lr * self.gradient(X, Y) / X.shape[0]
    #         self.theta -= step
    #         self.errors.append(self.loss(X, Y))
    #         if np.linalg.norm(step) <= THRESHOLD:
    #             break

    def gradient(self, X, Y):
        return X.T @ (sigmoid(X @ self.theta) - Y)

    def predict(self, X):
        padded_X = self.pad_with_ones(X)
        return (sigmoid(padded_X @ self.theta) > 0.5).astype('int')

    def pad_with_ones(self, X):
        return np.column_stack((np.ones((X.shape[0], 1)), X))


def demo():
    file = datautil.get_data_file_path('Social_Network_Ads.csv')

    data = pd.read_csv(file, )
    data = data.replace({r'\n': ''}, regex=True)
    X = data.values[1:, 2:-1].astype('int')
    Y = data.values[1:, -1:].astype('int')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    plt.plot(model.errors)
    plt.show()

    confusion = confusion_matrix(Y_test, Y_pred)
    print(confusion)


if __name__ == '__main__':
    demo()