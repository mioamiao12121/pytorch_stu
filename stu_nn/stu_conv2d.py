import  torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter


dataset=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,
                                     transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(dataset,batch_size=64)

class base_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)


    def forward(self,input):
        output=self.conv1(input)
        return output

base_model=base_model()
writer=SummaryWriter("../dataset_logs__nn")
step=0
for data in data_loader:
    imgs,targets=data
    outputs=base_model(imgs)
    print(imgs.shape)
    print(outputs.shape)
    writer.add_images("conv2d_1",imgs,step)
    outputs=torch.reshape(outputs,(-1,3,30,30))
    # tensorboard输出必为3通道，通道数不符会报错，通过使用reshape将其变为3通道，-1是让其自动匹配应该的参数
    writer.add_images("conv2d_2",outputs,step)
    step=step+1
writer.close()