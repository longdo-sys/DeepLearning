import torch
from torch import nn
from torch.nn import init
import numpy as np
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

# mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/', train=True, download=False,
#                                                 transform=transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/', train=False, download=False,
#                                                transform=transforms.ToTensor())

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # 输出层是全连接层，所以直接用一个线性模块就可以了
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


# net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 定义模型
from collections import OrderedDict

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# softmax 和交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
