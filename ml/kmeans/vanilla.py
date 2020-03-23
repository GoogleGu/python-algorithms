import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util


class KMeans:

    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = [None for _ in range(k)]
        self.classes = [[] for i in range(self.k)]

    def fit(self, X):
        for i in range(self.k):
            self.centroids[i] = X[i]

        for i in range(self.max_iterations):
            self.classes = [[] for i in range(self.k)]

            for sample in X:
                distances = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(sample)

            old_centroids = self.centroids.copy()
            for i in range(len(self.classes)):
                self.centroids[i] = np.average(self.classes[i], axis=0)

            for i in range(len(old_centroids)):
                if np.linalg.norm(old_centroids[i]-self.centroids[i]) > self.tolerance:
                    break
                return self.classes, self.centroids

        return self.classes, self.centroids

    def plot(self):
        colors = 10*["r", "g", "c", "b", "k"]
        for centroid in self.centroids:
            plt.scatter(centroid[0], centroid[1], s=130, marker="x")
        for i in range(self.k):
            color = colors[i]
            for sample in self.classes[i]:
                plt.scatter(sample[0], sample[1], color=color,s=5)
        plt.show()


if __name__ == '__main__':
    data_file = util.get_data_file_path("knn.csv")
    X = pd.read_csv(data_file).values[:, :2]

    kmeans = KMeans()
    classes, centroids = kmeans.fit(X)
    print(centroids)
    kmeans.plot()
