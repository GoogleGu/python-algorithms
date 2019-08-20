import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from util import get_data_file_path

data = get_data_file_path('kmeans-data.csv')
# import data into CountVectorizer 或 TFIDFVectorizer
df = pd.read_csv(data)

df.rename(column={'v1': 'Label', 'v2': 'Text'}, inplace=True)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
Y = df.numLabel

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# train naive bayes classifier
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred, labels=[0, 1])
