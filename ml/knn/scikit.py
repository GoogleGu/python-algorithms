import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from util import get_data_file_path


data_path = get_data_file_path('knn.csv')
data = pd.read_csv(data_path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
