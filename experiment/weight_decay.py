# 数据样本特征的维度为p 使用线性函数 y = 0.05 + ∑i=1 p 0.01xi+ϵ

# 高维线性回归实验
import numpy.random
import torch
import torch.nn as nn
import numpy as np
import sys

from torch import float32

sys.path.append("..")
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(numpy.random.normal(0, 0.01, size=labels.size()), dtype=float32)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# 从零开始实现权重衰减
# 初始化模型参数

def init_param():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2范数惩罚项 这里只惩罚模型的权重参数。
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    # 初始化模型
    w, b = init_param()
    # 定义存放损失函数结果的数组
    train_ls, test_ls = [], []
    # 开始进行周期性循环
    for _ in range(num_epochs):
        # 迭代中取出X, y
        for X, y in train_iter:
            # 构建损失函数，注意要添加惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            # 通过sum将l转化成标量？
            l.sum()
            # w 和 b梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            # 损失函数求导
            l.backward()
            # 优化函数
            d2l.sgd([w, b], lr, batch_size)
        # 添加结果到train_ls, test_ls中
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    # 作图 并 输出结果
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    d2l.plt.show()
    print('l2 norm of w:', w.norm().item(), 'l2 norm of b:', b.norm().item())


fit_and_plot(3)
