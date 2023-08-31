"""
优化器：
for input, target in dataset:
    optimizer.zero_grad() # 每一步都需要将梯度清零
    output = model(input) # 通过网络
    loss = loss_fn(output, target)  # 计算出误差
    loss.backward() # 将误差进行反向传播
    optimizer.step() # 调用优化器，每一步的参数都得到优化
"""
import torch
import torchvision.datasets
from torch import nn
from  torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential,CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader


class loss_nn(nn.Module):
    """
    神经网络的搭建，包括卷积、池化、摊平、线性
    """
    def __init__(self):
        super(loss_nn, self).__init__()
        self.moudle1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x=self.moudle1(x)
        return x

loss_nn=loss_nn()
dataset=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,download=True
                                     ,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(dataset,batch_size=64)

loss=nn.CrossEntropyLoss() # 损失函数

optim=torch.optim.SGD(loss_nn.parameters(),lr=0.01)
# 优化器，lr为学习速率，太大不稳定，太小速度慢

step=0
for epoch in range(20): #从每一轮看损失函数的变化
    running_loss=0
    for data in data_loader:
        imgs,target=data
        outputs=loss_nn(imgs)
        reult_loss=loss(outputs,target)
        optim.zero_grad() # 梯度调为0
        reult_loss.backward()   # 反向传播计算梯度
        optim.step() # 对每一个参数进行调优
        running_loss+=reult_loss
    print("神经网络输出与真实值之间的差",running_loss)