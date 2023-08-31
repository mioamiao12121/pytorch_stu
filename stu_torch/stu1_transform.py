"""
tensor数据类型:tensor包装了神经网络中需要的一些参数，包括后向传播的层数、使用设备类型

Tensor如何使用
Image.open()  # 读取图片路径
"""
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2

imge_path= "../hymenoptera_data/train/bees/16838648_415acd9e3f.jpg"
img=Image.open(imge_path)

writer=SummaryWriter("../logs")

trans_tensor=transforms.ToTensor()
tensor_img=trans_tensor(img)

writer.add_image("tensor_img",tensor_img)

cv2_img=cv2.imread(imge_path)

writer.close()