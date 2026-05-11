import os
import json
import random
import sys
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 配置Matplotlib中文显示
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 数据集读取与分割：按比例划分训练集/验证集，生成类别索引映射
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    root = "/home/nvidia/Desktop/Ch3/sample"
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    
    all_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    all_class.sort()
    class_indices = {k: v for v, k in enumerate(all_class)}
    
    with open('class_indices.json', 'w') as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=4)
    
    train_images_path, train_images_label = [], []
    val_images_path, val_images_label = [], []
    supported = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg']
    
    for cla in all_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        random.shuffle(images)
        val_num = int(len(images) * val_rate)
        val_path = images[:val_num]
        train_path = images[val_num:]
        
        train_images_path.extend(train_path)
        train_images_label.extend([class_indices[cla]] * len(train_path))
        val_images_path.extend(val_path)
        val_images_label.extend([class_indices[cla]] * len(val_path))
    
    print(f"Total images: {len(train_images_path)+len(val_images_path)}")
    print(f"Train samples: {len(train_images_path)}, Val samples: {len(val_images_path)}")
    return train_images_path, train_images_label, val_images_path, val_images_label

# 自定义数据集类：加载图片数据并应用变换
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert('RGB')
        label = self.images_class[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 自定义批处理：堆叠图片张量，转换标签为张量
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

# 卷积神经网络模型：2层卷积+池化 + 2层全连接（含Dropout防过拟合）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 前向传播：卷积->激活->池化 重复 -> 展平 -> 全连接 -> Dropout -> 输出
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 训练函数：单轮训练，返回平均损失和准确率
def trainer(model, optimizer, data_loader, epoch, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(data_loader, file=sys.stdout)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / labels.size(0)
        total_loss += loss.item()
        total_acc += acc
        pbar.set_description(f"[Epoch {epoch}] Loss: {loss.item():.3f} Acc: {acc:.3f}")
    
    return total_loss/len(data_loader), total_acc/len(data_loader)

# 评估函数：验证集评估，返回整体准确率
def evaluator(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
    
    return total_correct / len(data_loader.dataset)

# 可视化预测结果：展示指定数量样本的真实标签与预测标签
def visualize_predictions(model, data_loader, device, class_names, num_samples=4):
    model.eval()
    
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    plt.figure(figsize=(12, 8))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 2, i+1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * 255
        img = img.astype(np.uint8)
        
        plt.imshow(img)
        plt.title(f"True: {class_names[labels[i].item()]}, Pred: {class_names[preds[i].item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 超参数配置
    data_root = "./data"
    batch_size = 16
    epochs = 10
    lr = 0.001
    img_size = 128

    # 数据加载与预处理
    train_paths, train_labels, val_paths, val_labels = read_split_data(data_root)
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = MyDataSet(train_paths, train_labels, transform)
    val_dataset = MyDataSet(val_paths, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=val_dataset.collate_fn)

    # 模型初始化（优先使用CUDA）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环：记录训练损失/准确率、验证准确率
    train_losses, train_accs, val_accs = [], [], []
    for epoch in range(epochs):
        train_loss, train_acc = trainer(model, optimizer, train_loader, epoch, device)
        val_acc = evaluator(model, val_loader, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        train_accs.append(train_acc)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), "animal_classifier.pth")
    print("模型已保存为 animal_classifier.pth")

    # 可视化训练过程：损失/训练准确率/验证准确率曲线
    plt.figure(figsize=(15, 4))
    
    plt.subplot(131)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(range(1, epochs+1), train_accs, label='Train ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.subplot(133)
    plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 可视化预测结果
    visualize_predictions(model, val_loader, device, class_names)