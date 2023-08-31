"""
正则化层：可以加快神经网络的计算速度
flatten摊平
损失函数：计算实际输出和目标之间的差距，为我们更新输出提供一定的依据(反向传播）
优化器：根据梯度调整参数，降低误差（TORCH.OPTIM)
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear

class linear_nn(nn.Module):
    """
    神经网络使用最大池化，返回输出
    保留特征
    """
    def __init__(self):
        super(linear_nn, self).__init__()
        self.linear=Linear(196608,10)

    def forward(self,input):
        output=self.linear(input)
        return output



dataset=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)
# 若drop_last 采用默认值会出现报错，会产生 点数不匹配

linear_nn=linear_nn()
# writer=SummaryWriter("pool_logs")
step=0
for data in dataloader:
    imgs,targets=data
    # writer.add_images("linear_input",imgs,step)
    outputs=torch.flatten(imgs) # 摊平,功能与outputs=torch.reshape(imgs,(1,1,1,-1))一致
    outputs=linear_nn(outputs)#  每个批次的样本数1，深度1，高度1，宽度自动计算，-1出现在哪里那里的值就是自动计算，而不是被赋值为1
    # writer.add_images("linear_show",outputs,step)
    print(imgs.shape)
    print(outputs.shape)
    step=step+1
# writer.close()

