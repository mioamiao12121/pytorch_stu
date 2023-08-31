import torch
import torchvision
import torch.nn as nn

vgg16_true=torchvision.models.vgg16(pretrained=True)
vgg16_false=torchvision.models.vgg16(pretrained=False)
print(vgg16_true)

train_data=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=True,download=True,
                                        transform=torchvision.transforms.ToTensor())

vgg16_true.add_module('add_linear',nn.Linear(1000,10)) # 在现有的模型上新增加一层
print(vgg16_true)

vgg16_false.classifier[6]=nn.Linear(4096,10) # 将模型的参数进行修改
print(vgg16_false)

# 模型的保存1，模型结构和参数
torch.save(vgg16_true,"保存路径")

# 保存2，保存参数
torch.save(vgg16_true.state_dict(),"保存路径") # 转换 成字典格式，保存了模型的参数

# 模型的加载1
model1=torch.load("模型保存路径")

# 模型的加载
vgg16_flase=torchvision.models.vgg16(pretrained=False)# 先加载没有参数的模型
vgg16_false.load_state_dict("保存路径") # 再加载模型的参数

# 用自己的网络模型需要把模型类的定义再复制到保存操作之前，否则可能会报错
