from torch.utils.data import Dataset
from PIL import Image
import os

class mydata(Dataset):

        def  __init__(self,root_dir,label_dir):
            self.root_dir=root_dir
            self.label_dir=label_dir
            self.path=os.path.join(self.root_dir,self.label_dir) # 将两个路径进行连接，增加代码的通用性
            self.img_path=os.listdir(self.path) # 将图片路径和标签整合成一个字典，便于信息的读取

        def __getitem__(self, index):
            img_name=self.img_path[index]
            img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
            img=Image.open(img_item_path)
            label=self.label_dir
            return img,label

        def __len__(self):
            return len(self.img_path)


root_dir = "/hymenoptera_data/train"  # 绝对路径用\\进行区分
label_dir = "/hymenoptera_data/train/ants"
label1_dir = "/hymenoptera_data/train/bees"
# path=os.path.join(root_dir,label_dir)  # 将两个路径链接起来
ants_dataset=mydata(root_dir,label_dir)
bee_dataset=mydata(root_dir,label1_dir)
img,label=ants_dataset[1] # 读取图片的顺序不是按照文件夹中的顺序，是按照图片名称的由小到大
img1,label1=bee_dataset[1]
img.show()
img1.show()

train_dataset=ants_dataset+bee_dataset # 将两个数据集进行拼接，可以在特定情况下完成数据集的仿照