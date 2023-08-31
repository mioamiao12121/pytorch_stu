import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from class_model import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 查看数据集的大小
test_data_size = len(test_data)
train_data_size = len(train_data)
print("训练集测试长度：{}".format(test_data_size))
print("测试集测试长度：{}".format(train_data_size))

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建模型
class_model = sqen_nn()

# 创建损失函数
loss_nn = nn.CrossEntropyLoss()

# 优化器
optim = torch.optim.SGD(class_model.parameters(), lr=0.01)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 3
# 总体的正确率
total_accurrcy = 0

writer = SummaryWriter("model_logs")
for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))
    for data in train_loader:
        imgs, targets = data
        outputs = class_model(imgs)
        loss = loss_nn(outputs, targets)

        # 优化器调参数
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    total_test_loss = 0
    with torch.no_grad():  # 测试集不需要梯度进行测试和优化
        for data in test_loader:
            imgs, targets = data
            outputs = class_model(imgs)
            loss = loss_nn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accurrcy = (outputs.argmax(1) == targets).sum()
            total_accurrcy = total_accurrcy + accurrcy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集的正确率：{}".format(total_accurrcy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1
writer.close()

torch.save(class_model, "basic_model1.pth")
