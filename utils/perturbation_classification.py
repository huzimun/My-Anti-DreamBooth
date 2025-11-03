from torchvision import models, transforms
from torch import nn, optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

# 加载预训练的 EfficientNet 或 ResNet
model = models.efficientnet_b0(pretrained=True)
# 替换分类头为二分类
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# 冻结特征提取层
for param in model.features.parameters():
    param.requires_grad = False

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 图像输入大小，EfficientNet-B0 默认输入尺寸为 224x224
image_size = 224

# 定义训练集的数据增强和验证集的数据预处理
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet 均值
                         [0.229, 0.224, 0.225])   # ImageNet 方差
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset_path = "/data1/humw/Codes/My-Anti-DreamBooth/data/perturbed_images_dataset_v5"
output_path = "/data1/humw/Codes/My-Anti-DreamBooth/data/perturbed_images_dataset_v5"

# 数据集路径
train_dataset = datasets.ImageFolder(dataset_path + '/train', transform=train_transforms)
val_dataset = datasets.ImageFolder(dataset_path + '/val', transform=val_transforms)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def test_model(model, val_loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, output_path, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        # 每轮在验证集上测试
        val_loss, val_acc = test_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
    # 存储losses和accs到文件，直接写入txt文件   
    losses_file = os.path.join(output_path, 'losses.txt')
    accs_file = os.path.join(output_path, 'accs.txt')
    np.savetxt(losses_file, np.array([train_losses, val_losses]), delimiter=',', header='train_loss,val_loss', comments='')
    np.savetxt(accs_file, np.array([train_accs, val_accs]), delimiter=',', header='train_acc,val_acc', comments='')
    # 绘制曲线并保存，使用不同线型和标记，便于黑白打印和视觉障碍人群区分
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, 'o-', label='Train Loss', color='black', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, 's--', label='Val Loss', color='green', marker='s', linestyle='--')
    plt.plot(epochs, train_accs, '^-.', label='Train Acc', color='blue', marker='^', linestyle='-.')
    plt.plot(epochs, val_accs, 'D:', label='Val Acc', color='orange', marker='D', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training and Validation Loss/Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, './train_val_curve.png'))  # 保存到当前路径
    plt.show()
    return train_losses, train_accs, val_losses, val_accs

# 用法示例
train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, output_path, num_epochs=500)
val_loss, val_acc = test_model(model, val_loader, criterion)