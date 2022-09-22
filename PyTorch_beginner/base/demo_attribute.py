import torch
# 资源的合理调度也是训练效率的一部分
dev = torch.device("cpu")
# 只有一块显卡
dev = torch.device("cuda")
a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)


# 稀疏和低秩是重要的概念
# 稀疏可使模型变得非常简单，对数据稀疏化表示能够减少开销
indices = torch.tensor([[0, 1, 1], [2, 0, 3]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)

sp = torch.sparse_coo_tensor(indices, values, [2, 4])
den = sp.to_dense()
# 稀疏转稠密并输出
print(sp)
print(den)

b = torch.tensor([1, 2, 3], dtype=torch.float32, device=torch.device("cuda"))

