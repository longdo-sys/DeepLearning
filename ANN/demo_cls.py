import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as util_data
from simple_CNN import CNN
# data
train_data = dataset.MNIST(root="mnist",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)
# batchSize 方法 -- 数据量过大时内存cpu等负载过重
train_load = util_data.DataLoader(dataset=train_data,
                                  batch_size=64,
                                  shuffle=True)
test_load = util_data.DataLoader(dataset=test_data,
                                 batch_size=64,
                                 shuffle=True)


# net
# class CNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         # 这些部分的数字都是怎么定的呢 全连接层最后那块怎么算的呢
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.fc = torch.nn.Linear(14 * 14 * 32, 10)
#
#     def forward(self, x):
#         out = self.conv(x)
#         # 此处的out.size()[0], -1 还要在看看
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         return out


cnn = CNN()
# 设备变成GPU
cnn = cnn.cuda()
# loss
loss_func = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# train
for epoch in range(10):
    for i, (images, label) in enumerate(train_load):
        images = images.cuda()
        label = label.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch {} iter {}/{}, loss_train: {}".format(epoch+1, i, len(train_data) // 64, loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, label) in enumerate(test_load):
        images = images.cuda()
        label = label.cuda()
        outputs = cnn(images)
        # 此处的label 大小为 batchsize
        # outputs 为 batchsize * 10
        loss_test += loss_func(outputs, label)
        _, pred = outputs.max(1)
        # item()的作用
        accuracy += (pred == label).sum().item()
    # 为什么一个除64 另一个不用呢
    loss_test = loss_test / (len(test_data) // 64)
    accuracy = accuracy / (len(test_data))
    print("epoch {}, loss_test: {}, accuracy: {}".format(epoch+1, loss_test.item(), accuracy))
# save
torch.save(cnn, "model/mnist_model.pkl")
