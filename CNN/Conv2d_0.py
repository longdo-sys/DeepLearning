import numpy as np
import torch
from torch import nn

'''
边缘检测：
pytorch《动手学深度学习》5.1
时隔一个暑假，再次接触CNN，先做个最简单的边缘检测

'''


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# K = torch.tensor([[0, 1], [2, 3]])
# print(corr2d(X, K))


# net
class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        # 此处的nn.Parameter作用？ 貌似是存储参数，但还有其他手段达到这个效果吗，怎么实现的呢
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 准备数据
X = torch.ones(6, 8)
X[:, 2:6] = 0

# 手动实验的话
K = torch.tensor([[1, -1]])

Y = corr2d(X, K)
print(Y)

# 通过数据学习到核数组
conv2d = Conv2d(kernel_size=(1, 2))

step = 50
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()  # 这里要用sum() 否则不好求导？
    # backward 又是什么来着？
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1) % 5 == 0:
        print("%d 时，loss值为：%.3f" % (i+1, l))

print(conv2d.weight, conv2d.bias)

