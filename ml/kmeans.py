import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from cv2 import imread, imshow, imwrite
from sklearn.cluster import KMeans

from lib import datautil


data_file = datautil.get_data_file_path("knn.csv")


def sk_usage():
    """
    使用scikit learn完成kmeans算法的实现
    """
    img = imread(data_file.as_posix())

    pixel = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    print(pixel.shape)
    pixel_new = deepcopy(pixel)

    print(img.shape)

    model = KMeans(n_clusters=5)
    # 注意，KMeans的fit_predict方法入参必须是两个dimension，d1是各个样本，d2是每个样本的features, 返回值是每个样本分到第几类的列表
    labels = model.fit_predict(pixel)
    # cluster_centers_返回各类中心点的列表
    palette = model.cluster_centers_

    print(labels)

    for i in range(len(pixel)):
        pixel_new[i, :] = palette[labels[i]]

    imwrite('zipped.jpg', np.reshape(pixel_new, (img.shape[0], img.shape[1], 3)))


class VanillaKMeans:

    """
    从零实现二维空间的kmeans
    """

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


def demo():
    X = pd.read_csv(data_file).values[:, :2]
    kmeans = VanillaKMeans()
    classes, centroids = kmeans.fit(X)
    print(centroids)
    kmeans.plot()


class Solution:
    """
    用kmeans算法解决LeetCode 1478题
    https://leetcode-cn.com/problems/allocate-mailboxes/
    """
    def minDistance(self, houses, k):
        return min(self.kmeans(houses, k) for i in range(50))

    def kmeans(self, houses, k):
        """

        Args:
            houses: houses[i] 是第 i 栋房子在一条街上的位置
            k: 有多少个邮筒需要安排

        Returns:
            每栋房子与离它最近的邮筒之间的距离的最小总和
        """

        def distance(x1, x2):
            return abs(x1-x2)

        # 初始化k个起始中心点
        mailboxes = [random.choice(houses) for _ in range(k)]

        # 建立一个list存储每个house的所属类别，一开始先全部分到第一类中
        classes = [0 for _ in houses]

        # 开始迭代
        cost = 1
        while cost > 0:
            # 分配点到各个类中
            for i in range(len(houses)):
                center_distances = [distance(houses[i], mailbox) for mailbox in mailboxes]
                classes[i] = center_distances.index(min(center_distances))

            # 变动中心点
            cost = 0
            for i in range(k):
                class_i = [houses[j] for j in range(len(houses)) if classes[j] == i]
                if not class_i:
                    continue
                new_pos = int(np.median(class_i))
                cost += abs(new_pos - mailboxes[i])
                mailboxes[i] = new_pos

        # 计算总距离
        total_dist = 0
        for i in range(len(houses)):
            total_dist += distance(houses[i], mailboxes[classes[i]])
        print(mailboxes)
        return total_dist

for i in range(10):
    print(Solution().minDistance([48,43,20,18,6,5,35,41,1,2,27,17,37], k=7))
