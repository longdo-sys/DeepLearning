import torch
from torch import nn
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

'''
多输入通道和多输出通道：
pytorch《动手学深度学习》5.3


'''


def corr2d_multi_in(X, K):
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    # 形状的问题在corr2d中已经解决了，此处关心的只是有几个channel
    for i in range(1, K.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(X.shape)


#
# print(corr2d_multi_in(X, K))

# ##多输出通道
def corr2d_multi_in_out(X, K):
    # ♥每个输出通道上的结果由卷积核在该输出通道上的核数组与整个输入数组计算而来
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# K在原来的基础上加了一维，实现了多(3)输出通道
K = torch.stack([K, K + 1, K + 2])
print(K.shape)

print(corr2d_multi_in_out(X, K))


# ##1×1卷积层  1×1卷积层通常用来调整网络层之间的通道数，并控制模型复杂度
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)


X = torch.randn((3, 3, 3))
K = torch.randn((2, 3, 1, 1))
Y1 = corr2d_multi_in_out(X, K)

Y2 = corr2d_multi_in_out_1x1(X, K)

print((Y1 - Y2).norm().item() < 1e-6)
