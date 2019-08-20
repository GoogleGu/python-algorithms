
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import get_data_file_path
from ml.models import LinearModel


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class LogisticRegression(LinearModel):

    def __init__(self, shape):
        super().__init__(shape)

    def loss(self, X, Y):
        temp = np.log(sigmoid(X @ self.theta))
        m = X.shape[0]
        return np.sum(-(Y.T @ temp + (1-Y).T @ (1 - temp)) / (2*m))

    def gradient(self, X, Y):
        return X.T @ (sigmoid(X @ self.theta) - Y)

    def predict(self, X):
        padded_X = self.pad_with_ones(X)
        return (sigmoid(padded_X @ self.theta) > 0).astype('int')


file = get_data_file_path('Social_Network_Ads.csv')

data = pd.read_csv(file, )
data = data.replace({r'\n': ''}, regex=True)
X = data.values[1:, 2:-1].astype('int')
Y = data.values[1:, -1:].astype('int')
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression((3, 1))
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.plot(model.errors)
plt.show()

confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)

# ERROR：有问题，loss为负值，总是会把结果全部判为True
