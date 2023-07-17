import torch

a = torch.tensor([[1, 2], [2, 4], [3, 6]])
b = torch.tensor([[1, 1], [1, 1], [1, 1]])
print(a.shape)

print(torch.tensordot(a, b, dims=2))
