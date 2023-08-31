"""
卷积提取特征，池化压缩特征
池化层分为最大池化层和平均池化层
最大池化层是只最大池化时最大的值，池化默认不重合（不设置步长的情况下），保留边缘特征，使得进行训练的参数变小，提高训练速度
最大池化的作用是保留数据特征，减少数据量
平均池化：保留背景特征
ceiling是保留多的数
floor是舍去多的数，不保留，默认不保留
reshape 中的-1 能自动调整前面的维度
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d

class pool_nn(nn.Module):
    """
    神经网络使用最大池化，返回输出
    保留特征
    """
    def __init__(self):
        super(pool_nn, self).__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool(input)
        return output


# region 简单池化
# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]],dtype=torch.float) # 数据一般用浮点型，用torch.float进行改变数据类型
# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)
#
# pool_nn=pool_nn()
# output=pool_nn(input)
# print(output)
# endregion

# region 图像池化
dataset=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
pool_nn=pool_nn()
writer=SummaryWriter("pool_logs")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("pool_input",imgs,step)
    outputs=pool_nn(imgs)
    writer.add_images("pool_show",outputs,step)
    step=step+1
writer.close()

# endregion