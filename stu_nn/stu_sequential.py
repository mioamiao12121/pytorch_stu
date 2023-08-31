import torch
from torch import nn
from  torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential


class sqen_nn(nn.Module):
    """
    神经网络的搭建，包括卷积、池化、摊平、线性
    """
    def __init__(self):
        super(sqen_nn, self).__init__()
        # self.conv1=Conv2d(3,32,5,padding=2) # 输出通道数等于卷积核数，与输入通道数无关
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2=MaxPool2d(2)
        # self.conv3=Conv2d(32,64,5,padding=2)
        # self.maxpool3=MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)
        # sequential 将不同的操作写在一个模块里
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
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x = self.maxpool2(x)
        # x=self.conv3(x)
        # x = self.maxpool3(x)
        # x=self.flatten(x)
        # x=x.view(x.size(0),-1)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.moudle1(x)
        return x

sqen_nn=sqen_nn()
print(sqen_nn)
input=torch.ones((64,3,32,32))
output=sqen_nn(input)
print(output.shape)