# 处理过拟合的常用方法，权重衰减，丢弃法
import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

# 高维线性回归实验

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 从零开始实现

