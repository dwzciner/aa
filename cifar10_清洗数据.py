import os
import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# 下载和加载 CIFAR-10 数据集（如果你还没有下载）
cifar10 = datasets.CIFAR10(root='data', train=True, download=True)

# 获取图像和标签
images, labels = cifar10.data, torch.tensor(cifar10.targets)

# 创建训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 保存路径
train_root = './data/cifar10/train'
val_root = './data/cifar10/val'

# 创建文件夹
for i in range(10):
    os.makedirs(os.path.join(train_root, str(i)), exist_ok=True)
    os.makedirs(os.path.join(val_root, str(i)), exist_ok=True)

# 保存训练集图像
for i, (image, label) in enumerate(zip(train_images, train_labels)):
    label = label.item()
    image_path = os.path.join(train_root, str(label), f'{i}.jpg')  # 保存路径
    image = Image.fromarray(image)  # 将图像转换为 PIL 格式
    image.save(image_path)

# 保存验证集图像
for i, (image, label) in enumerate(zip(val_images, val_labels)):
    label = label.item()
    image_path = os.path.join(val_root, str(label), f'{i}.jpg')  # 保存路径
    image = Image.fromarray(image)  # 将图像转换为 PIL 格式
    image.save(image_path)

print("CIFAR-10 数据已成功转换并保存到按标签分类的文件夹中！")
