"""
非线性激活函数：relu、sigmoid
relu负数部分为0，非零部分不变
"""
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class relu1(nn.Module):

    def __init__(self) :
        super().__init__()
        self.relu=torch.nn.ReLU() # 激活函数可以改变为sigmoid等函数

    def forward(self,input):
        output=self.relu(input)
        return  output

# region 简单relu
# input=torch.tensor([[1,0.5],
#                     [-1,3]])
# input=torch.reshape(input,(-1,1,2,2))
# print(input.shape)
# relu1=relu1()
# output=relu1(input)
# print(output)
# endregion

# region 图像relu
dataset=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
relu1=relu1()
writer=SummaryWriter("pool_logs")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("relu_input",imgs,step)
    outputs=relu1(imgs)
    writer.add_images("relu_show",outputs,step)
    step=step+1
writer.close()

# endregion
