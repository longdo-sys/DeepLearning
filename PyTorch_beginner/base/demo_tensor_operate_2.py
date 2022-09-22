# 切片

import torch


a = torch.rand(3, 4)

out = torch.chunk(a, 2, dim=0)
print(out)
for t in out:
    print(t.shape)

out = torch.split(a, 3, dim=0)
print(out)

out = torch.split(a, [1, 2], dim=0)
print(out)

# 变形操作
print("reshape")
a = torch.rand(2, 3)
out = torch.reshape(a, (3, 2))
print(a)
print(out)
# 转置
print("t")
print(torch.t(out))

print("transpose")
print(torch.transpose(out, 0, 1))

print("a变成三维")
a = torch.rand(1, 2, 3)
print(a)
out = torch.squeeze(a)
# print(a.shape)
# print(a)
print(out)
print(out.shape)

print("unsqueeze")
out = torch.unsqueeze(a, dim=-1)
print(out.shape)
# unbind后得到的也是元组， 和切片相似
# unbind去除某个维度
out = torch.unbind(a, dim=0)
print(out)

# flip操作
out = torch.flip(a, dims=[1, 2])
print(out)

# 填充
out = torch.full((2, 3), 10)
print(out)
