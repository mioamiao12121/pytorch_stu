"""
dataset 告诉我们数据集在什么位置
dataloader 加载器，把数据加载到神经网络中
"""
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_trans=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_trans,download=True )
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_trans,download=True )

print(test_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
writer=SummaryWriter("../dataset_logs")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("dataset_picture",img,i)
writer.close()