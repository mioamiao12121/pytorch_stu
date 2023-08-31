import torch
from PIL import Image
import torchvision.transforms
from keras.models import Sequential
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

image_path = "./img.png"
image = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)


class sqen_nn(nn.Module):
    """
    神经网络的搭建，包括卷积、池化、摊平、线性
    """

    def __init__(self):
        super(sqen_nn, self).__init__()
        self.moudle1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.moudle1(x)
        return x


# 如果gpu上训练的模型要对应到CPU上，需要增加map_location=torch.device('cpu')，即
# model1_base = torch.load("basic_model1.pth",map_location=torch.device('cpu'))

model1_base = torch.load("basic_model1.pth")
print(model1_base)

image = torch.reshape(image, (1, 3, 32, 32))
model1_base.eval()
with torch.no_grad():
    output = model1_base(image)
print(output)
print(output.argmax(1))
