
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_FILE = '/Users/arthur/code/source/algorithm/python-algorithms/dataset/airfoil_self_noise.dat'


class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, features, labels):
        self.data = [(feature, label) for feature, label in zip(features, labels)]

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


def get_data_ch7(data_file):
    data = np.genfromtxt(data_file, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    features = torch.tensor(data[:1500, :-1], requires_grad=True)
    labels = torch.tensor(data[:1500, -1], requires_grad=True)
    return features, labels


def linear_reg(x, w, b):
    return x.double() @ w.double() + b


def train_ch7(optimizer, features, labels, batch_size=1, num_epochs=10):
    loss, f = nn.MSELoss(), linear_reg
    # 初始化参数
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32), requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def total_loss():
        predictions = f(features, w, b)
        return loss(predictions, labels.view(-1, 1)).mean().item()

    ls = [total_loss()]
    loader = DataLoader(CustomDataSet(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        for batch_i, (X, y) in enumerate(loader):
            # 反向传播求导
            loss_val = loss(f(X, w, b), y.view(-1, 1)).mean()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            loss_val.backward()
            # 根据导数调整模型参数
            optimizer.descent([w, b])
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(total_loss())
    return ls


class Grad:

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.states = None
        self.eps = 1e-6

    def _init_states(self, params):
        pass

    def descent(self, params):
        for p in params:
            p.data -= self.hyperparams['lr'] * p.grad.data


class MomentumGrad(Grad):

    def _init_states(self, params):
        v_w = torch.zeros((params.shape[1], 1))
        v_b = torch.zeros(1)
        self.states = (v_w, v_b)

    def descent(self, params):
        if self.states is None:
            self._init_states(params)
        for p, v in zip(params, self.states):
            v.data = self.hyperparams['momentum'] * v + self.hyperparams['lr'] * p.grad.data
            p.data -= v.data
            p.data -= self.hyperparams['lr'] * p.grad.data


class AdaGrad(Grad):

    def _init_states(self, params):
        s_w = torch.zeros((params.shape[1], 1))
        s_b = torch.zeros(1)
        self.states = (s_w, s_b)

    def descent(self, params):
        if self.states is None:
            self._init_states(params)
        for p, s in zip(params, self.states):
            s += (p.grad.data ** 2)
            p.data -= self.hyperparams['lr'] * p.grad / torch.sqrt(s + self.eps)


class RMSProp(AdaGrad):

    def descent(self, params):
        if self.states is None:
            self._init_states(params)
        gamma = self.hyperparams['gamma']
        for p, s in zip(params, self.states):
            s = gamma * s + (1 - gamma) * (p.grad.data ** 2)
            p.data -= self.hyperparams['lr'] * p.grad.data / torch.sqrt(s + self.eps)


class AdaDelta(AdaGrad):

    def _init_states(self, params):
        s_w, s_b = torch.zeros((features.shape[1], 1)), torch.zeros(1)
        delta_w, delta_b = torch.zeros((features.shape[1], 1)), torch.zeros(1)
        self.states = (s_w, delta_w), (s_b, delta_b)

    def descent(self, params):
        if self.states is None:
            self._init_states(params)
        rho = self.hyperparams['rho']
        for p, (s, delta) in zip(params, self.states):
            s = rho * s + (1 - rho) * (p.grad.data ** 2)
            g = p.grad.data * torch.sqrt((delta + self.eps) / (s + self.eps))
            p.data -= g.data
            delta = rho * delta + (1 - rho) * (g.data ** 2)


class Adam(AdaGrad):

    def _init_states(self, params):
        v_w, v_b = torch.zeros((features.shape[1], 1)), torch.zeros(1)
        s_w, s_b = torch.zeros((features.shape[1], 1)), torch.zeros(1)
        self.states = (v_w, s_w), (v_b, s_b)

    def descent(self, params):
        if self.states is None:
            self._init_states(params)
        beta1, beta2 = 0.9, 0.999
        for p, (v, s) in zip(params, self.states):
            v = beta1 * v + (1 - beta1) * p.grad.data
            s = beta2 * s + (1 - beta2) * p.grad.data ** 2
            v_bias_corr = v / (1 - beta1 ** self.hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** self.hyperparams['t'])
            p.data -= self.hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + self.eps)
            self.hyperparams['t'] += 1


if __name__ == "__main__":
    features, labels = get_data_ch7(DATA_FILE)
    optimizer = Adam(hyperparams={'lr': 0.01, 'gamma': 0.9, 'rho': 0.99, 't': 1})
    num_epochs = 5
    ls = train_ch7(optimizer, features, labels, batch_size=50, num_epochs=num_epochs)
    # 打印结果和作图
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
