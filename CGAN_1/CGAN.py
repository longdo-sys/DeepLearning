# %%
import argparse
import os
import numpy as np
import math
from torch.utils.data import TensorDataset as TD
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader as DL
import torch.nn as nn
import torch.nn.functional as F
import torch

# %%
# 定义程序用到的参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")  # 训练循环数
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")  # 小批次数据量
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")  # 学习速率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")  # 噪声向量维度
parser.add_argument("--n_classes", type=int, default=44, help="number of classes for dataset")  # 条件向量维度（种类数）
parser.add_argument("--data_size", type=int, default=4, help="size of each image dimension")  # 生成数据的维度
opt = parser.parse_args(args=[])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_shape = (opt.data_size)  # 生成数据的维度：4


# %%
# 定义生成器和判别器的网络结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 定义模块网络
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat, momentum=0.5))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # 定义网络结构
        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 288),
            *block(288, 256),
            *block(256, 64),
            nn.Linear(64, opt.data_size)
        )

    # 前向传播函数
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1)
        img = self.model(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义网络结构
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + 4, 256),
            nn.BatchNorm1d(256, momentum=0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum=0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img, labels), -1)
        validity = self.model(d_in)
        return validity


# %%
# 加载真实数据并分批
# max_rows是提取的数据条数，调试时取500，下同
data = np.loadtxt(r".\Real_data_Train_5_3dBm.csv", delimiter=",", dtype=np.float32)
#data = np.loadtxt(r"C:\Users\Desktop\CGAN数据\real_data1_pam8_filter.csv", delimiter=",", dtype=np.float32)
temp_data = torch.from_numpy(data)  # 维度为4
data_DL = DL(temp_data, batch_size=opt.batch_size, shuffle=False, drop_last=True)
# 加载条件向量（标签）并分批
c = np.loadtxt(r".\Condition_vector_Train_5_3dBm.csv", delimiter=",", dtype=np.float32)
#c = np.loadtxt(r"C:\Users\Desktop\CGAN数据\condition_vector1_pam8_filter.csv", delimiter=",", dtype=np.float32)
c_data = torch.from_numpy(c)  # 维度为44
c_DL = DL(c_data, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# %%
# 定义loss函数和优化器
# 判别器Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00001)
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

for m in generator.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.detach())
        m.bias.detach().zero_()
for m in discriminator.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.detach())
        m.bias.detach().zero_()

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
# 定义两个列表，保存训练时loss值以便于画图
DLoss = []
GLoss = []


# %%
# 保存loss图片
def plotSaveTrainLoss(epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(GLoss, label="G")
    plt.plot(DLoss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./%d.png" % epoch)
    plt.show()


# %%
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (real_data, labels) in enumerate(zip(data_DL, c_DL)):
        batch_size = real_data.shape[0]
        real_data = real_data.cuda()
        labels = labels.cuda()
        # Adversarial ground truths
        valid = torch.rand(batch_size, 1).cuda() * 0.5 + 0.7
        fake = torch.rand(batch_size, 1).cuda() * 0.3

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z = torch.randn(batch_size, opt.latent_dim).cuda()
        # gen_labels = Variable(FloatTensor(np.random.uniform(-1,1,(batch_size,opt.n_classes))))
        gen_labels = torch.randn(batch_size, opt.n_classes).cuda()
        # gen_labels = labels
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for r in range(1):
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_data, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            #d_loss = d_real_loss + d_fake_loss
            DLoss.append(d_loss.item())
            d_loss.backward()
            optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        for r in range(1):
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, opt.latent_dim).cuda()
            # gen_labels = Variable(FloatTensor(np.random.uniform(-1,1,(batch_size,opt.n_classes))))
            # gen_labels = torch.randn(batch_size,opt.n_classes).cuda()
            # gen_imgs = generator(z, gen_labels)
            gen_real_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_real_imgs, labels)
            # g_loss = adversarial_loss(validity, valid)
            L1 = ((real_data - gen_real_imgs).norm(1, 1)).mean()
            # L1 = adversarial_loss(real_data , gen_real_imgs)
            g_loss = adversarial_loss(validity, valid)*10 + 0.2*L1 #加入正则项后的G_loss
            #g_loss = adversarial_loss(validity, valid) + 0.1*L1  # 加入正则项后的G_loss
            g_loss.backward()
            optimizer_G.step()
        GLoss.append(g_loss.item())
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f][L1: %f]"
        % (epoch, opt.n_epochs, d_loss.item(), g_loss.item(), L1.item())
    )
    if (epoch % 100 == 0):
        plotSaveTrainLoss(epoch)
        torch.save(discriminator, '.\D_8.19_2000epoch_5_3dBm.pth')
        torch.save(generator, '.\G_8.19_2000epoch_5_3dBm.pth')


