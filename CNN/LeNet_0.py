import time
import torch
from torch import nn, optim
import sys
import numpy

sys.path.append('..')
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
LeNet：
pytorch《动手学深度学习》5.5


'''


# LeNet网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # 输出个数分别是 120， 84 和 10

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
            # Sigmoid作用是什么来着，为什么这里不要了呢
            # nn.Sigmoid()
        )

    # conv和fc为什么能够连上呢， 或者说能连上的条件是什么呢
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
# print(net())

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# 评估精确度 和softmax那时候比 加上了device相关的处理
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定device就用net中的
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式将会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义模式
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


# 对train_ch3函数略作修改，确保计算使用的数据和模型同在内存或显存上
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("train on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            # 这里这一堆 .sum() .cpu() .item() 都是些什么意思呢，顺序能调吗
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


lr, num_epochs = 0.01, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

# 展示，代码竟然还能够使用。。。
for i, (images, label) in enumerate(test_iter):
    images = images.cuda()
    label = label.cuda()
    outputs = net(images)
    _, pred = outputs.max(1)
    #     # item()的作用
    #     accuracy += (pred == label).sum().item()
    #
    #     # 想要获得图片先要转化到cpu上来
    images = images.cpu().numpy()
    label = label.cpu().numpy()
    pred = pred.cpu().numpy()
    #     # batchSize * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        #         # 这里的transpose 是怎么回事呢
        #         # 三个维度；相当于有X轴，Y轴，Z轴 transpose（1，0，2）表示X轴与Y轴发生变换之后；
        im_data = im_data.transpose(1, 2, 0)
        lab_data = label[idx]
        pred_data = pred[idx]
        #
        print("label: ", lab_data)
        print("pred:", pred_data)
#         cv2.imshow("im_data", im_data)
#         cv2.waitKey(0)
