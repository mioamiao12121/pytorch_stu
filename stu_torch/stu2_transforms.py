from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

write=SummaryWriter("/logs")
img=Image.open("../hymenoptera_data/train/ants/6240338_93729615ec.jpg")
print(img)

trans_tensor=transforms.ToTensor()
img_tensor=trans_tensor(img) # 这里输入的数据类型是原来的img
write.add_image("totensor",img_tensor)
print(img_tensor[0][0][0])


# 标准化归一化,由于图像是RGB三通道，所以需要三个均值和标准差
trans_norm=transforms.Normalize([0.1,0.1,0.1],[0.1,0.1,0.1])
img_norm=trans_norm(img_tensor) # 这里需要的数据类型是tensor
write.add_image("norm1",img_norm)
print(img_norm[0][0][0])

# resize将图片进行大小的转换
print(img.size)
trans_resize=transforms.Resize((512,512))
# 图片类型转换： PIL->RESIZE->totensor
img_resize=trans_resize(img)
img_resize=trans_tensor(img_resize)
write.add_image("resize",img_resize)
print(img_resize)

# compose将Tensorflows的两个函数封装在一个compose函数里面
trans_compose=transforms.Compose([trans_norm,trans_resize])  # 将要结合的两个操作放在一个中括号里，compose只接受一个参数
img_compose=trans_compose(img_tensor)
write.add_image("compose",img_compose)

# randomcrop 随机裁剪
trans_rand=transforms.RandomCrop(30)
trans_compose2=transforms.Compose([trans_rand,trans_tensor])
for i in range(10):
    img_crop=trans_compose2(img)
    write.add_image("randomcrop",img_crop,i)

write.close()
"""
tensor 里的call函数可以直接调用类对象，不用.来调用方法
Ctrl+p 提示需要的参数
totensor将图片类型转换成Tensor类型
"""
