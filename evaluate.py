import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim
import time
import os

from mec.builder import MEC  # 你的MEC模型

# CIFAR-10 数据集相关配置
def load_cifar10_data(batch_size=128):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# 评估代码
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 加载模型并恢复checkpoint
def load_model(checkpoint_path, model, device):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{checkpoint_path}'")
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
    return model

def main():
    checkpoint_path = 'checkpoint_0060.pth.tar'  # 你的checkpoint路径
    batch_size = 5
    model_arch = 'resnet18'  # 可根据需要调整模型架构
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载CIFAR-10数据
    train_loader, test_loader = load_cifar10_data(batch_size)

    # 创建MEC模型并加载checkpoint
    model = MEC(models.__dict__[model_arch], dim=512, pred_dim=128)  # 你的模型初始化
    model = model.to(device)
    model = load_model(checkpoint_path, model, device)

    # 在测试集上评估模型
    accuracy = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
