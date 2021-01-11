import torch
import numpy as np

'''
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
'''

# LR

'''
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

def forward(x):
	y = w *x +b
	return y

x = torch.tensor(2)
print(forward(x))
'''

from torch.nn import Linear
torch.manual_seed(1)


model = Linear(in_features=1, out_features=1)
print(model.bias, model.weight)

x = torch.tensor([2.0])
print(model(x))




