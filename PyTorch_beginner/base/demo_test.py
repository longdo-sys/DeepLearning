import torch
'''
几种特殊的Tensor
'''
a = torch.Tensor(2, 3)
print(a)
print(a.type())

b = torch.Tensor((3, 2))
print(b)
print(b.type())
# 一样的大小但都用0来填充
c = torch.zeros_like(a)
print(c)
print(c.type())
d = torch.eye(3, 3)
print(d)
print(d.type())
'''
随机
'''
print('''随机''')
a = torch.rand(2, 2)
print(a)
print(a.type())
# 正态分布 经常用于初始化参数
a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())
# 均匀分布
a = torch.Tensor(2, 2).uniform_()
print(a)
print(a.type())
'''
序列
'''
print('''
序列
''')
a = torch.arange(0, 11, 2)
print(a)
print(a.type())
# 从 s 到 e 均匀切分成 n 份
# 拿到等间隔的n个数字
a = torch.linspace(2, 10, 3)
print(a)
print(a.type())
b = a
# 随机排列 可以打乱数据
a = torch.randperm(10)
# a = torch.randperm(10)
print(a)
print(a.type())
