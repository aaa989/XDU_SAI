import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 全局配置
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Inception模块：多分支卷积结构（1x1、3x3、5x5、池化）
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))
        branch3x3 = F.relu(self.branch3x3_3(branch3x3))
        
        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.branch5x5_2(branch5x5))
        
        branch1x1 = F.relu(self.branch1x1(x))
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.relu(self.branch_pool(branch_pool))
        
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

# 主网络：卷积 + Inception + 全连接分类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(88, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = self.mp(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 单轮训练函数
def train(epoch, model, trainloader, criterion, optimizer, batch_loss_list, epoch_loss_list):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        if batch_idx % 300 == 299:
            avg_batch_loss = running_loss / 300
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_batch_loss:.4f}')
            batch_loss_list.append(avg_batch_loss)
            running_loss = 0.0
    epoch_avg_loss = total_loss / len(trainloader)
    epoch_loss_list.append(epoch_avg_loss)

# 测试集评估函数
def test(model, testloader, epoch_acc_list):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    epoch_acc_list.append(acc)
    print(f'Accuracy on test set: {acc:.2f}%\n')

# 预测结果可视化
def detection_visualization(model, testloader, classes):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    img_show = images.cpu() / 2 + 0.5
    img_np = img_show.numpy().transpose(0, 2, 3, 1)

    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Model Detection Visualization: True Label vs Predicted Label", fontsize=16)
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img_np[i])
        true_name = classes[labels[i]]
        pred_name = classes[preds[i]]
        ax.set_title(f"True:{true_name}\nPred:{pred_name}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# 训练损失+测试准确率曲线
def draw_final_curve(epoch_loss_list, epoch_acc_list):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    epochs = list(range(1, len(epoch_loss_list)+1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'red'
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Average Training Loss', color=color1)
    ax1.plot(epochs, epoch_loss_list, color=color1, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color2)
    ax2.plot(epochs, epoch_acc_list, color=color2, marker='s', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Inception Model Training Loss and Test Accuracy on CIFAR10')
    fig.tight_layout()
    plt.show()

# 批次损失变化曲线
def draw_process_loss(batch_loss_list):
    plt.figure(figsize=(10,4))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.plot(batch_loss_list, color='orange')
    plt.xlabel('Training Batch Interval (per 300 batches)')
    plt.ylabel('Batch Average Loss')
    plt.title('Real-time Training Batch Loss Changing Curve')
    plt.grid(True)
    plt.show()

# 主程序入口
if __name__ == '__main__':
    # 数据预处理与加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # 模型初始化
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 训练过程记录
    batch_loss_list = []
    epoch_loss_list = []
    epoch_acc_list = []

    # 训练循环（50轮）
    for epoch in range(50):
        train(epoch, model, trainloader, criterion, optimizer, batch_loss_list, epoch_loss_list)
        test(model, testloader, epoch_acc_list)

    # 结果可视化
    draw_process_loss(batch_loss_list)
    draw_final_curve(epoch_loss_list, epoch_acc_list)
    detection_visualization(model, testloader, classes)