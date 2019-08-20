import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from util import get_data_file_path

data_file = get_data_file_path('studentscores.csv')
df = pd.read_csv(data_file)
X, Y = df.values[:, 0:1], df.values[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


plt.scatter(X, Y, marker='X')
plt.plot(X_test, Y_pred)
plt.show()
