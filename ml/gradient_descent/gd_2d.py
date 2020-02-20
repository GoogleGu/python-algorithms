import math

import numpy as np
import torch
import matplotlib.pyplot as plt


def train_2d(trainer, init=[-5, -2], lr=0.1, n=20):
    x1, x2 = init
    v1, v2, s1, s2 = 0, 0, 0, 0
    results = [(x1, x2)]
    for i in range(1, n + 1):
        x1, x2, v1, v2, s1, s2 = trainer(x1, x2, v1=v1, v2=v2, s1=s1, s2=s2, lr=lr, n=i)
        results.append((x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd(x1, x2, v1, v2, s1, s2, lr, n):
    return x1 - lr * 0.2 * x1, x2 - lr * 4 * x2, 0, 0, 0, 0


def momentum_gd(x1, x2, v1, v2, s1, s2, lr, n, gamma=0.5):
    v1 = gamma * v1 + lr * 0.2 * x1
    v2 = gamma * v2 + lr * 4 * x2
    return x1 - v1, x2 - v2, v1, v2, 0, 0


def adagrad(x1, x2, v1, v2, s1, s2, lr, n):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= lr / math.sqrt(s1 + eps) * g1
    x2 -= lr / math.sqrt(s2 + eps) * g2
    return x1, x2, 0, 0, s1, s2


def rmsprop(x1, x2, v1, v2, s1, s2, lr, n, gamma=0.9):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * (g1 ** 2)
    s2 = gamma * s1 + (1 - gamma) * (g2 ** 2)
    x1 -= lr / math.sqrt(s1 + eps) * g1
    x2 -= lr / math.sqrt(s2 + eps) * g2
    return x1, x2, 0, 0, s1, s2


def adam(x1, x2, v1, v2, s1, s2, lr, n):
    g1, g2, eps, beta1, beta2 = 0.2 * x1, 4 * x2, 1e-6, 0.9, 0.999
    v1 = beta1 * v1 + (1 - beta1) * g1
    v2 = beta1 * v2 + (1 - beta1) * g2
    s1 = beta2 * s1 + (1 - beta2) * g1 * g1
    s2 = beta2 * s2 + (1 - beta2) * g2 * g2
    v1_hat = v1 / (1 - beta1 ** n)
    v2_hat = v2 / (1 - beta1 ** n)
    s1_hat = s1 / (1 - beta2 ** n)
    s2_hat = s2 / (1 - beta2 ** n)
    x1 -= lr * v1_hat / (math.sqrt(s1_hat) + eps)
    x2 -= lr * v2_hat / (math.sqrt(s2_hat) + eps)
    return x1, x2, v1, v2, s1, s2


if __name__ == "__main__":
    gd_result = train_2d(adam, lr=0.4, n=100)
    show_trace_2d(f_2d, gd_result)
    plt.show()
