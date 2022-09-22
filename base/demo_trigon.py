import numpy as numpy
import torch

a = torch.zeros(2, 3)
b = torch.cos(a)
c = int(torch.sum(a))
print(a)
print(b)
