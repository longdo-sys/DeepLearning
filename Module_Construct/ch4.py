from collections import OrderedDict

import torch
from torch import nn


# Module类是所有神经网络模块的基类
# Module类 是一个可供自由组建的部件，既可以时一个层, 又可以是一个模型
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)


# 当前向计算为简单串联各个层的计算
# 它可以接受一个子模块的有序字典(OrderedDict)
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        # 如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)
print(net(X))
