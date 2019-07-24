import os
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import DATA_ROOT

# 导入数据
data = pd.read_csv(DATA_ROOT + os.sep + 'kmeans-data.csv')
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))


# 初始化K均值算法的k个中心
k = 3
C_x = np.random.randint(0, np.max(X), size=k)
C_y = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)


def distance(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)


clusters = np.zeros(len(X))
error = 1000

# error为0表示中心点没有变动
while error != 0:
    # 将点分组
    for i in range(len(X)):
        distances = distance(X[i], C)
        cluster_index = np.argmin(distances)
        clusters[i] = cluster_index
    # 存储之前的中心点
    C_old = deepcopy(C)
    # 计算各组新的中心点
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = distance(C, C_old, None)

# 可视化各个组
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()
