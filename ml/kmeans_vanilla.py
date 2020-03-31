import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util


class KMeans:

    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = None
        self.classes = None
        self.X = None

    def fit(self, X):
        self.X = X
        self.centroids = X[:self.k]
        self.classes = np.array([0 for _ in range(len(X))])

        for i in range(self.max_iterations):
            # distances = []
            # for centroid in self.centroids:
            #     distances.append(np.linalg.norm(X-centroid, axis=1))
            # self.classes = np.argmax(np.array(distances), axis=0)

            for i in range(len(X)):
                distances = [np.linalg.norm(X[i] - centroid) for centroid in self.centroids]
                self.classes[i] = distances.index(min(distances))

            old_centroids = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = np.average(self.X[self.classes == i], axis=0)

            for i in range(len(old_centroids)):
                if np.linalg.norm(old_centroids[i]-self.centroids[i]) > self.tolerance:
                    break
                return self.classes, self.centroids

        return self.classes, self.centroids

    def plot(self):
        colors = 10*["r", "g", "c", "b", "k"]
        for centroid in self.centroids:
            plt.scatter(centroid[0], centroid[1], s=130, marker="x")
        for i in range(len(self.X)):
            plt.scatter(self.X[i][0], self.X[i][1], color=colors[self.classes[i]],s=5)
        plt.show()


if __name__ == '__main__':
    data_file = util.get_data_file_path("knn.csv")
    X = pd.read_csv(data_file).values[:, :2]

    kmeans = KMeans()
    classes, centroids = kmeans.fit(X)
    print(centroids)
    kmeans.plot()
