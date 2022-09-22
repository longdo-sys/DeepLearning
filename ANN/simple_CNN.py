import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 这些部分的数字都是怎么定的呢 全连接层最后那块怎么算的呢
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        # 此处的out.size()[0], -1 还要在看看
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
