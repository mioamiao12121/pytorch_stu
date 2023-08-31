import  torch
from torch.nn import L1Loss

input=torch.tensor([1,2,3],dtype=float)
target=torch.tensor([1,2,5],dtype=float)

input=torch.reshape(input,(1,1,1,3))
target=torch.reshape(target,(1,1,1,3))
# 神经网络一般使用浮点数
loss=L1Loss()
result=loss(input,target)

print(result)