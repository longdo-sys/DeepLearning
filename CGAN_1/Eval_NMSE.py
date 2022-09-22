import numpy as np
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义模块网络
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat,momentum=0.5))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        # 定义网络结构
        self.model = nn.Sequential(
            *block(54, 288),
            *block(288, 256),
            *block(256, 64),
            nn.Linear(64, 4)
        )
    # 前向传播函数
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1)
        img = self.model(gen_input)
        return img

G = Generator()
#G = torch.load('D:\gsr\G.pth')
G = torch.load(r'.\G_8.19_2000epoch_5_3dBm.pth')
G.cuda()

temp_data = np.loadtxt(r".\Real_data_Test_5_3dBm.csv",delimiter=",",dtype = np.float32,max_rows=20000)
#temp_data = np.loadtxt(r"C:\Users\Desktop\CGAN数据\real_data1_pam8_filter.csv",delimiter=",",dtype = np.float32,max_rows=200)
real_data = torch.from_numpy(temp_data).cuda()#维度为4
c_data = np.loadtxt(r".\Condition_vector_Test_5_3dBm.csv",delimiter=",",dtype = np.float32,max_rows=20000)
#c_data = np.loadtxt(r"C:\Users\Desktop\CGAN数据\condition_vector1_pam8_filter.csv",delimiter=",",dtype = np.float32,max_rows=200)
c = torch.from_numpy(c_data).cuda()#维度为44
z = torch.randn(20000, 10).cuda()

gen_data = G(z,c)
d1 = torch.sum(torch.pow(gen_data - real_data,2))
d2 = torch.sum(torch.pow(real_data,2))
MSE_err = d1/d2
print(MSE_err)
#p.savetxt(r"C:\Users\Desktop\CGAN生成数据\噪声向量_PAM8_OAM_2_1_32GPAM8_512000_5000epoch.csv",z.cpu(), delimiter=",")
#np.savetxt(r"C:\Users\Desktop\OAM3_GAN\6\SC_TEST_6dBm.csv",gen_data.detach().cpu().numpy(), delimiter=",")




