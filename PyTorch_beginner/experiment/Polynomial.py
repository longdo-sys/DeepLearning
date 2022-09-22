# %matplotlib inline
import torch
import numpy as np
import sys

from torch import float32

sys.path.append("../..")
import d2lzh_pytorch as d2l

# 生成数据集 y=1.2x−3.4x^2+5.6x^3+5+ϵ,
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))  # features就是x向量

ploy_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)  # cat起拼接作用

labels = (true_w[0] * ploy_features[:, 0] + true_w[1] * ploy_features[:, 1] + true_w[2] * ploy_features[:, 2] + true_b)

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=float32)  # labels那么多服从的正态分布的数作为ϵ


# 定义、训练和测试模型

# 定义作图函数semilogy

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


# 训练与测试步骤
num_epochs, loss = 100, torch.nn.MSELoss()


# 定义拟合函数
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # shape[0]理解为第一维，shape[1]理解为第二维 而shape[-1]指的是向量的列数
    net = torch.nn.Linear(train_features.shape[-1], 1)

    # 形成train_iter

    batch_size = min(10, train_labels.shape[0])
    # 这里.data是啥作用？
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for i in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train_loss', train_ls[-1], 'final epoch: test_labels', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias', net.bias.data)


# 正常  [1.2, -3.4, 5.6], 5
# fit_and_plot(ploy_features[:n_train, :], ploy_features[n_train:, :], labels[:n_train], labels[n_train:])

# 欠拟合
# fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])

# 训练样本不足(过拟合)
fit_and_plot(ploy_features[0:2, :], ploy_features[n_train:, :], labels[0:2], labels[n_train:])

