import torch
from torch import float32

'''
房价预测 保存好后模型的加载
'''

# data
import numpy as np
import re

ff = open("data/housing.data")
data = []
for item in ff:
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

X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]


# 需要自动定义好网络，不能直接load
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


net = torch.load("model/model.pkl")
loss_func = torch.nn.MSELoss()

x_data = torch.tensor(X_test, dtype=float32)
y_data = torch.tensor(Y_test, dtype=float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.01
print("loss_test {}".format(loss_test))
