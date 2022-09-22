import torch
# broadcast 广播机制
'''
这样看来python的广播机制其实并不是很强，总体来说就是 右对齐，之后依次考察各个维度，如果该维度上的数值相同或者其中一个为1或为none，
则可以进行广播运算否则会出错，运算结果的维度取该维度上数值较大的值
'''
a = torch.rand(2, 1)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)

a = torch.rand(2, 3)
b = torch.rand(3)  # 1,3
c = a + b
print(a)
print(b)
print(c)

print("-"*30)

a = torch.rand(4, 5)
b = torch.rand(1)
c = a + b
print(c)


import numpy as np

np.nan

