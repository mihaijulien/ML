import torch
import numpy as np

x = torch.rand(5, 3)
print(x)

x = torch.tensor([5.5,3])
print(x)
x = torch.tensor([[1,2],[3,4]])
print(x)

data = np.array([[1,2],[3,4]])
x = torch.tensor(data)
print(x)

x = torch.ones(5)
print(x)