import time
import math

import seaborn as sns
import pandas as pd
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt


def gd(df, x, lr=0.01, n=20):
    """
    梯度下降演示
    Params
        df: f的导数函数
        x: x的初始值
    """
    results = [(x, lr)]
    for _ in range(n):
        x = x - lr * df(x)
        results.append((x, lr))
    print("After {} iterations, x = {}".format(n, x))
    return results


def quantile_plot(seq, **kwargs):
    n = max(abs(min(seq)), abs(max(seq)), 1)
    f_line = np.arange(-n, n, 0.1)
    plt.plot(f_line, [x ** 2 for x in f_line])
    plt.plot(seq, [x ** 2 for x in seq], marker='o', color='orange')


if __name__ == "__main__":
    data = []
    data.extend(gd(lambda x: 2 * x, x=1, lr=1.01, n=20))
    data.extend(gd(lambda x: 2 * x, x=1, lr=0.4, n=20))
    data.extend(gd(lambda x: 2 * x, x=1, lr=0.04, n=20))

    df = pd.DataFrame(data, columns=['x', 'lr'])
    graph = sns.FacetGrid(df, col="lr")
    graph.map(quantile_plot, "x")
    plt.show()
