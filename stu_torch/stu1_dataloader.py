import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_data=torchvision.datasets.CIFAR10("E:\\python\\虚拟环境\\测试36\\dataset",train=False,
                                       transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
# dataset:加载数据集，batch_size:每次加载多少个样本，shuffle：洗牌，使数据每次都重新洗牌（打乱）
# num_work:加载进程的个数（在windows下大于0可能会出现错误，出现brokenpiperror报错，
# drop_last：如果数据最后有余数，舍去或者不舍去

img,target=test_data[0] # 测试集第一张图片

# 将打包后的图片进行遍历
writer=SummaryWriter("../dataset_logs")
step=0
for data in test_loader:
    imgs,targets=data
    print(imgs.shape)
    print(targets)
    writer.add_images("ceshi",imgs,step)
    # 加载单个图片用writer.add_image，加载多个图片用writer.add_images
    step=step+1
writer.close()
# 运行结果：
# torch.Size([4, 3, 32, 32])，4个样本，3 通道颜色，32x32的大小
# tensor([7, 0, 1, 1])，将4个样本的target打包成一个数组，每个样本的target为7/0/1/1