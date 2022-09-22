import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

from torch import float32

sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# 使用pandas读取文件
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

print(train_data.shape)
print(test_data.shape)
# loc中数据时列名，是字符串，前后都要取；iloc中数据是int整型，是python默认的前闭后开
# iloc[ : , : ] 行列切片以“，”隔开，前面的冒号就是取行数，后面的冒号是取列数
# print(train_data.iloc[:, 1:-1])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features)

# 预处理数据
# 连续数值的特征做标准化，缺失的数据值？
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转成指示特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape 特征值从79增加到了 331
# 通过values属性得到numpy格式的数据，并转成tensor方便后面的训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=float32)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=float32).view(-1, 1)

# 训练模型
loss = torch.nn.MSELoss()


def get_net(feature_num):
    net = torch.nn.Linear(feature_num, 1)

    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# 定义比赛用来评价模型的对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


# 使用Adam优化算法优化，优点：对学习率相对不那么敏感
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# k折交叉验证 其用来选择模型设计并调节超参数; 下面函数返回第i折交叉验证时所需要的训练和验证数据
# k折交叉验证：改善训练数据不够用的问题
# 原始训练数据集拆成k个不重合的子数据集，做k次模型训练和验证
# 每次使用一个子数据集验证模型，并使用其他k-1个子数据集训练模型
# 所以 k次训练和验证中每次用来训练和验证模型的子数据集都不同；
# 最后，对k次训练误差和验证误差分别求平均

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.concat((X_train, X_part), dim=0)
            y_train = torch.concat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 训练k次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
            d2l.plt.show()
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


# 模型选择
# 使用一组未经调优的超参数并计算交叉验证误差，
# 可改动这些超参数来尽可能减少平均测试误差
k, num_epochs, lr, weight_decay, batch_size = 5, 220, 5, 0, 60
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


# 预测并在Kaggle提交结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
