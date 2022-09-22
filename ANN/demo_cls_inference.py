import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as util_data
from simple_CNN import CNN
import numpy
import cv2
# data
test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)
# batchSize 方法 -- 数据量过大时内存cpu等负载过重
test_load = util_data.DataLoader(dataset=test_data,
                                 batch_size=64,
                                 shuffle=True)


# net 不引入CNN会报错，实际上还是使用了
cnn = torch.load("model/mnist_model.pkl")
# 设备变成GPU
cnn = cnn.cuda()

accuracy = 0
for i, (images, label) in enumerate(test_load):
    images = images.cuda()
    label = label.cuda()
    outputs = cnn(images)
    _, pred = outputs.max(1)
    # item()的作用
    accuracy += (pred == label).sum().item()

    # 想要获得图片先要转化到cpu上来
    images = images.cpu().numpy()
    label = label.cpu().numpy()
    pred = pred.cpu().numpy()
    # batchSize * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        # 这里的transpose 是怎么回事呢
        # 三个维度；相当于有X轴，Y轴，Z轴 transpose（1，0，2）表示X轴与Y轴发生变换之后；
        im_data = im_data.transpose(1, 2, 0)
        lab_data = label[idx]
        pred_data = pred[idx]

        print("label: ", lab_data)
        print("pred:", pred_data)
        cv2.imshow("im_data", im_data)
        cv2.waitKey(0)
# 为什么一个除64 另一个不用呢
accuracy = accuracy / (len(test_data))
print("accuracy: {}".format(accuracy))
