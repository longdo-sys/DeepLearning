import torch
from torch import float32

'''
房价预测
'''

# data
import numpy as np
import re

ff = open("data/housing.data")
data = []
for item in ff:
    # 为什么加一个 结果差异挺大  这里还有很多不懂的
    # 函数re.sub(pattern,repl,string,count,flags)，
    # pattern:表示正则表达式中的模式字符串；
    # repl:被替换的字符串（既可以是字符串，也可以是函数）；
    # string:要被处理的，要被替换的字符串；
    # count:匹配的次数
    # flags为标志位，用于控制正则表达式的匹配方式，如是否区分大小写，多行匹配等等
    #
    # strip()函数
    # 用来删除一些字符，把字符串中的一些字符替换成空，，
    # 括号中有东西的时候删除指定字符(当做一个个字符处理，删除力很强)，该方法存在限制：首位不能有空格，否则无法删除指定字符
    # 参数为空时（即括号里没东西）默认删除空白符（包括'\n', '\r', '\t', ' ')，删除开头和结尾的空白符
    # 两个版：lstrip()和rstrip(), l和r分别表示左和右
    out = re.sub(r"\s{2,}", " ", item).strip()
    # print(out)

    data.append(out.split(" "))
# np的相关知识要加强
data = np.array(data).astype(np.float64)
print(data.shape)
# 拿到了数据，关于切分的知识点看看
Y = data[:, -1]
X = data[:, 0:-1]
print(X.shape)
print(Y.shape)
'''
省略号的使用
'''
#
# a[:,1] = [2,5]
#
# a[...,1] = [2,5]
# 以上二者是等同的
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

# print(X_train.shape)
# print('Y', Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

# 关注一下欠拟合的解决方案

'''
'''


# net
class Net(torch.nn.Module):
    def __init__(self, features, n_output):
        super(Net, self).__init__()
        # 加入隐藏层使得网络结构更加复杂
        self.hidden = torch.nn.Linear(features, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss()
# optimizer
# 传入参数时使用初始化的网络.parameters()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
y_data = torch.zeros_like(torch.tensor(Y_train, dtype=float32))
pred = y_data
# i = 0
# loss = 0
for i in range(10000):
    # 这里需要处理下数据
    x_data = torch.tensor(X_train, dtype=float32)
    y_data = torch.tensor(Y_train, dtype=float32)
    # 用定义的网络前向运算
    pred = net.forward(x_data)
    # loss 注意得到的pre是第二维度为1的二维张量
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.01
    # 优化
    optimizer.zero_grad()
    loss.backward()
    # 优化参数
    optimizer.step()
    # loss本身太大 或者 学习率太大 或者 有非法值存在 都可能会使loss为none
    print("iter {}, loss_train {}".format(i, loss))
    print(y_data[0:10])
    print(pred[0:10])
    # test

    x_data = torch.tensor(X_test, dtype=float32)
    y_data = torch.tensor(Y_test, dtype=float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.01
    print("iter {}, loss_test {}".format(i, loss_test))

# 两种保存模型的方式
# 一 全部保存 占用资源多
torch.save(net, "model/model.pkl")
# torch.load("")
# 二 保存参数
# torch.save(net.state_dict(), "params.pkl")
