import torch

a = torch.rand(2, 3)
print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a))
print(torch.argmin(a, dim=0))
print(torch.argmax(a, dim=0))
# 标准差
print(torch.std(a))
# 方差
print(torch.var(a))
# 中间值
print(torch.median(a))
# 众数
print(torch.mode(a))

a = torch.rand(2, 2)
print(a)
print(torch.histc(a, 6, 0, 0))
# 只支持一维的tensor, bincount可用来统计某一类别样本的个数
a = torch.randint(0, 10, [10])
print(a)
# 分别统计0-9出现的频次
print(torch.bincount(a))
