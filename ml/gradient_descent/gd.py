import time
import math

import torch
import numpy as np
import torch
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss

DATA_FILE = '/Users/arthur/code/source/algorithm/python-algorithms/dataset/airfoil_self_noise.dat'



def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    net, loss = d2l.linreg, d2l.squared_loss
    # 初始化参数
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()
    ls = [eval_loss()]

    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 反向传播求导
            with autograd.record():
                mean_loss = loss(net(X, w, b), y).mean()
            mean_loss.backward()
            # 根据导数调整模型参数
            trainer_fn([w, b], states, hyperparams)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')


def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
        p -= v


def get_data_ch7():
    data = np.genfromtxt(DATA_FILE, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    features = nd.array(data[:1500, :-1])
    labels = nd.array(data[:1500, -1])
    return features, labels


if __name__ == "__main__":
    features, labels = get_data_ch7()
    train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.5}, features, labels)
