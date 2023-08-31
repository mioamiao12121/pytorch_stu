import torch
import torchvision.datasets
from torch import nn
from  torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential,CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter


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

loss=nn.CrossEntropyLoss()

# writer=SummaryWriter("pool_logs")
step=0
for data in data_loader:
    imgs,target=data
    outputs=loss_nn(imgs)
    reult_loss=loss(outputs,target)
    reult_loss.backward()   # 反向传播计算梯度
    print("神经网络输出与真实值之间的差",reult_loss)
    # writer.add_images("all_nn",outputs,step)
    step=step+1
# writer.close()