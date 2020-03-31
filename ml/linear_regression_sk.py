import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# from util import get_data_file_path

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
