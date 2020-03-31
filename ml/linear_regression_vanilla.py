
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from util import get_data_file_path
from ml.models import LinearModel


class LinearRegression(LinearModel):

    def __init__(self):
        super().__init__((2, 1))

    def loss(self, X, Y):
        return np.sum((X @ self.theta - Y) ** 2)

    def gradient(self, X, Y):
        return np.sum((X.T @ (X @ self.theta - Y)), axis=0)

    def predict(self, X):
        return self.pad_with_ones(X.copy()) @ self.theta


data_file = get_data_file_path('studentscores.csv')
df = pd.read_csv(data_file)
X, Y = df.values[:, 0:1], df.values[:, 1:]

model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

plt.scatter(X, Y, marker='X')
plt.plot(X, Y_pred)
plt.show()

plt.plot(model.errors)
plt.show()
