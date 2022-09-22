import torch
from torch import nn

'''
池化层：
缓解卷积层对位置的过度敏感性
pytorch《动手学深度学习》5.4

'''


# 池化层不需要具体的卷积核，只需要给出卷积核的大小即可
def pool2d(X, pool_size, mode="max"):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    # 最终的结果是Y的形状
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()

    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = pool2d(X, (2, 2))
print(Y)

print(pool2d(X, (2, 2), "avg"))

# ##填充和步幅
X = torch.arange(16, dtype=float).view(1, 1, 4, 4)
print(X)

pool2d = nn.MaxPool2d(3)
print(pool2d(X))
# padding=1,上下左右都补1 stride=2 纵向横向都走2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# padding(1, 2) 上下补1，左右补2；stride(2, 3) 纵向是2，横向走3
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=3)
print(pool2d(X))

# ##多输入与多输出通道 池化层不能改变通道数
X = torch.cat((X, X+1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))