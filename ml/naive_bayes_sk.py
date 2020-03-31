import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from util import get_data_file_path

# stopset = stopwords.words("english")

data = get_data_file_path('spam.csv')
df = pd.read_csv(data, encoding='iso-8859-1')

df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
Y = df.Label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# train naive bayes classifier
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(conf_mat)
