from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from util import get_data_file_path

# stopset = stopwords.words("english")


class NaiveBayesClassifier:

    def __init__(self):
        self.prior = None
        self.conditional = dict()
        self.labels = None
        self.min_prob = 0

    def fit(self, X, Y):
        total_samples = len(Y)

        X = X.tolist()
        Y = Y.tolist()
        # 计算先验概率
        label_count = Counter(Y)
        self.labels = list(label_count.keys())
        self.prior = {key: value/total_samples for key, value in label_count.items()}

        # 计算条件概率
        features_count = Counter(feature for sample in X for feature in sample)
        self.min_prob = 1 / len(features_count)

        # initialization
        for label in label_count.keys():
            self.conditional[label] = dict()
            for feature in features_count.keys():
                self.conditional[label][feature] = 0

        # count all instances
        for i, sample in enumerate(X):
            for feature in sample:
                self.conditional[Y[i]][feature] += 1

        # calculate probability
        for label in label_count.keys():
            for feature in features_count.keys():
                # laplace normalization
                self.conditional[label][feature] = (self.conditional[label][feature]+1) / (label_count[label]+len(features_count))

    def predict(self, X):
        X = X.tolist()
        predictions = []
        for sample in X:
            sample_pred = []
            for label in self.labels:
                product_of_cond = 1
                for feature in sample:
                    product_of_cond *= self.conditional[label].get(feature, self.min_prob)
                sample_pred.append(product_of_cond * self.prior[label])
            predicted_label = self.labels[np.argmax(np.array(sample_pred))]
            predictions.append(predicted_label)
        return predictions


data = get_data_file_path('bayes.csv')
df = pd.read_csv(data)

Y = df.iloc[:, 2].values
X = df.iloc[:, :2].values

model = NaiveBayesClassifier()
model.fit(X, Y)
y_pred = model.predict(np.array([[2, 'S']], dtype=np.object))
print(y_pred)
