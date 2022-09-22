# torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
# torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
# torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
# torchvision.utils: 其他的一些有用的方法。

# 获取数据集
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

from torch import float32

sys.path.append("../..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/', train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/', train=False, download=False,
                                               transform=transforms.ToTensor())

# feature 对应高和宽均为28像素的图像。 transforms.ToTensor()将每个像素转换为[0.0, 1.0]的32位浮点数

feature, label = mnist_train[0]


# print(feature.shape, label)  # Channel x Height x Width 数据集中是灰度图像，所以通道数为1


# 数据集中包含10个类别:t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）
# sneaker（运动鞋）、bag（包）和ankle boot（短靴）

# 将数值标签转换成文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker',
                   'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 在一行里画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # _表示忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# X, y = [], []
# for i in range(10):
#    X.append(mnist_train[i][0])
#    y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量

# 实践中数据读取经常是训练的性能瓶颈。PyTorch的DataLoader能很方便地使用多进程读取数据

# batch_size = 256
# if sys.platform.startswith("win"):
#     nums_workers = 0
# else:
#    nums_workers = 4

# print("工作进程：", nums_workers)
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=nums_workers)
# test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=nums_workers)

# 查看读取训练数据需要的时间
# start = time.time()
# for X, y in train_iter:
#     continue
# print("%.2f sec" % (time.time() - start))

# softmax从零开始实现
import numpy as np

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 初始化模型
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

# 设置模型梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 实现softmax运算
# 对多维Tensor按维度操作
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))

# 求样本在各个输出类别上的预测概率
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)  # 将每行数据加到一起？
    return X_exp / partition  # 这里应用了传播机制


# X = torch.randn((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))

# 定义模型
def net(X):
    # 这里的-1表示行和列颠倒
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])


# y行列颠倒且view处理后成了两个行向量，1代表调用者？
# 使得能够依次提取y_hat中每行以y的每个值为列坐标的数
# y_hat.gather(1, y.view(-1, 1))
# print(y.view(-1, 1))
# print(y_hat)
# print(y_hat.gather(1, y.view(-1, 1)))

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率 预测概率最大的类别与真实类别比较。准确率 = 正确预测数量 / 总预测数量
def accuracy(y_hat, y):
    # item()将一个标量Tensor转换成一个Python number
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += ((net(X).argmax(dim=1) == y).float().sum().item())
        n += y.shape[0]
    return acc_sum / n


# print(evaluate_accuracy(test_iter, net))

# 训练模型 使用小批量随机梯度下降优化模型损失函数
num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + "\n" + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[:12], titles[:12])
