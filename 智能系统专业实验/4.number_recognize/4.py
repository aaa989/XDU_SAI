import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 数据加载与预处理：MNIST数据集
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

# LeNet-5网络结构定义
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # 卷积层1 + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # 卷积层2 + 池化
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # 展平后接入全连接层
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型训练函数
def train(model, device, train_loader, optimizer, epoch, log_interval=10, epoch_loss_list=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_epoch_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_epoch_loss = total_epoch_loss / len(train_loader)
    epoch_loss_list.append(avg_epoch_loss)

# 模型测试函数
def test(model, device, test_loader, acc_list=None):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    acc_list.append(accuracy)
    return accuracy

# 主流程：初始化+训练+测试+可视化
def main():
    # 设备选择：优先GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据加载
    train_loader, test_loader = load_data()
    # 模型初始化
    model = LeNet5().to(device)
    # 优化器配置
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.5
    )
    
    # 训练参数
    epochs = 10
    log_interval = 10
    epoch_loss_list = []
    accuracy_list = []
    epoch_x = list(range(1, epochs + 1))
    
    # 训练循环
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, epoch_loss_list)
        accuracy = test(model, device, test_loader, accuracy_list)
        
        # 保存最优模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_lenet5_model.pth')
            print(f"保存最佳模型，准确率: {accuracy:.2f}%")
    print(f"训练完成！最佳测试准确率: {best_accuracy:.2f}%")

    # 可视化训练损失与测试准确率
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Training Average Loss', color=color1)
    ax1.plot(epoch_x, epoch_loss_list, color=color1, marker='o', label='Loss Curve')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color2)
    ax2.plot(epoch_x, accuracy_list, color=color2, marker='s', label='Accuracy Curve')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('LeNet-5 Training Loss and Test Accuracy Trend')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()