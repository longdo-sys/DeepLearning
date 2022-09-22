import torch
import numpy as np
import random

# y=Xw+b+ϵ
# 生成数据集
from torch import float32

input_nums = 2
input_examples = 1000
true_w = [4.2, -3.4]
true_b = 4.2
# 列，行？

features = torch.randn(input_examples, input_nums, dtype=float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 噪声项 ϵϵ 服从均值为0、标准差为0.01的正态分布
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=float32)


# print(features[0], labels[0])


# 目的在于读取随机小批量样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        # 从乱序的下标中取出batch_size个下标
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
    print(X, y)
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (input_nums, 1)), dtype=float32)
b = torch.zeros(1, dtype=float32)

# 开启求梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型
def linreg(X, w, b):
    # 矩阵乘法
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    # 注意返回的是向量
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):
    for param in params:
        # 更改时用.data，防止？？？
        param.data -= lr * param.grad / batch_size


# 训练模型
# 之前的问题：小批量疑似只训练了一部分(a batch_size)？ 为何对损失函数的返回值求和操作？

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        # 求导过程还不是太熟悉，再看看
        l.backward()
        sgd([w, b], lr, batch_size)

        # 每次进行一次清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
