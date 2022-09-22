import torch
from torch import nn

'''
填充和步幅
pytorch《动手学深度学习》5.2

总得来看，
填充可增加输出的宽和高，常用来使输出和输入具有相同的高和宽
步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的1/n（n为大于1的整数）
'''


def comp_conv2d(conv2d, X):
    # 这里的(1, 1)代表批量大小和通道数 均为1 ; 这一步还挺关键的 3D or 4D view成了 2D
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道;view保证只改变视角，不改变数据本身


# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, padding=1, kernel_size=(3, 3))

X = torch.rand((8, 8))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

'''

stride

'''
X = torch.rand((8, 8))
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=2)
Y = comp_conv2d(conv2d, X)
print(Y.shape)