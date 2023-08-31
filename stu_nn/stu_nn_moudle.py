import torch
import torch.nn as nn
import torch.nn.functional as F

# 卷积网络中需要四个参数，在参数形式不够时使用torch.reshape将其变为 四个参数
# stride 步长。padding进行填充，进行增补是增补一圈不是单独增补一行或一列
class base_model(nn.Module):

    def __init__(self):
        super().__init__()
        #self.conv2=nn.ConvTranspose2d(1,20,5)
        # in_channel 输入层数，out_channel 输出层数，
        # 多个out_channel是通过增加卷积核的数量得到的，将最后得到的每层叠加得到最后的输出，每个卷积核可能不一样


    def forward(self,input):
        output=input+1
        return output

base_model=base_model()
x=torch.tensor(1.0)
output=base_model(x)
print(output)
