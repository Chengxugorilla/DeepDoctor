import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import ReadLabel


class CTData(Dataset):
    def __init__(self, path):
        self.path = path
        self.img_names = os.listdir(path)
        self.labels = ReadLabel.labels[1400:1600]
        self.pool2d = nn.MaxPool2d(4, padding=0, stride=4)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_name = self.img_names[index]
        img_path = os.path.join(self.path, image_name)
        img = Image.open(fp=img_path, mode='r')
        image_tensor = transforms.ToTensor()(img).cuda()
        image_tensor =image_tensor[:, 930:1500, 280:830]
        image_tensor = self.pool2d(image_tensor)  # 先欠采样。如果放进模型再欠采样会爆显存（因为pytorch会自动记录梯度项）
        # image_tensor = torch.unsqueeze(transforms.ToTensor()(img), 0).cuda()
        # print(self.img_names)
        label = self.labels[index]
        sample = (image_tensor, label)
        # sample = {'image': image_tensor, 'label': label}
        return sample
