import torch
# tensor

# torch.where

a = torch.rand(4, 4)
b = torch.rand(4, 4)
c = torch.where(a > 0.5, a, b)
print(a, b, c)

# torch.index_select
print('torch.index_select'+'-'*20)
a = torch.rand(4, 4)
out = torch.index_select(a, dim=0, index=torch.tensor([0, 2, 1]))
print(a)
print(out, out.shape)

# torch.gather
print("torch.gather")
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
# dim=0, out[i, j, k] = input[index[i, j, k], j, k]
out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 2, 1],
                                                 [0, 2, 3, 2],
                                                 [1, 3, 2, 2]]))
print(out)
print(out.shape)


# torch.masked_select 输出向量
print("torch.masked_select")
a = torch.linspace(1, 16, 16).view(4, 4)
mask = torch.gt(a, 8)
print(mask)
out = torch.masked_select(a, mask)
print(out)

# torch.take
print('torch.take')
out = torch.take(a, torch.tensor([1, 15, 13, 2]))
print(out)

# torch.nonzero

a = torch.tensor([[0, 2, 3, 0], [0, 2, 0, 0], [1, 0, 0, 0]])
out = torch.nonzero(a)
print(out)

# 拼接与组合
print("torch.cat")
a = torch.rand(2, 4)
b = torch.rand(1, 4)
out = torch.cat((a, b), dim=0)
print(out)

print('torch.stack')
a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
print(a)
print(b)
out = torch.stack((a, b), dim=2)
print(out)
print(out.shape)

