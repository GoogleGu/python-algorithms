import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from lib import datautil


data_file = datautil.get_data_file_path('Social_Network_Ads.csv')


def sk_usage():
    data = pd.read_csv(data_file)
    data = data.replace({r'\n': ''}, regex=True)
    X = data.values[1:, 2:-1].astype('int')
    Y = data.values[1:, -1].astype('int')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LinearSVC()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    print(confusion)
