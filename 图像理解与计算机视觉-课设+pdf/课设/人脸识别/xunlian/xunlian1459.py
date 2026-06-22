import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# ==========================================
# 1. ArcFace 度量学习损失层
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  
        self.m = m  
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, labels):
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        output = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * output) + ((1.0 - one_hot) * cosine)
        output *= self.s  
        
        return output

# ==========================================
# 2. 网络结构：ResNet18 + 解冻两层 + Dropout (防过拟合)
# ==========================================
class LFWFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(LFWFeatureExtractor, self).__init__()
        # 加载 ImageNet 预训练模型
        self.backbone = models.resnet18(pretrained=True)
        
        # 解冻两层！让 layer3, layer4 以及全连接层(fc)都参与训练
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True 
                
        # 【修改点 1】：加入 Dropout 层，物理断除“死记硬背”
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5), # 50%的神经元随机失活
            nn.Linear(in_features, embedding_dim)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. LFW 数据集加载与增强 (防过拟合)
# ==========================================
class LFWDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx].astype(np.uint8)
        img_pil = transforms.ToPILImage()(img_np)
        if self.transform:
            img_tensor = self.transform(img_pil)
        return img_tensor, self.labels[idx]

def get_dataloaders(min_faces=5, batch_size=64):
    print(f"\n[1/3] 正在加载 LFW 数据集 (保留至少 {min_faces} 张照片的人物)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0)
    
    images = lfw.images * 255.0  
    labels = lfw.target
    num_classes = len(lfw.target_names)
    print(f"数据加载完成！共有 {len(images)} 张图片，包含 {num_classes} 个不同的人物。")
    print(f"💡 已启用：Dropout、权重衰减、早停法与随机擦除，全面对抗过拟合！")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 【修改点 2】：更强的数据增强 (增加旋转角度，并加入 RandomErasing 随机擦除)
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15), # 增加轻微旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)) # 50%概率在人脸上贴黑块遮挡
    ])

    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = LFWDataset(X_train, y_train, train_transform)
    test_dataset = LFWDataset(X_test, y_test, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes

# ==========================================
# 4. 可视化绘制函数 (英文标题)
# ==========================================
def plot_metrics(history):
    print("\n[3/3] 正在生成训练可视化图表...")
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
    plt.title('Training and Validation Loss (ArcFace)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_metrics_v2.png', dpi=300)
    print("-> 可视化图表已保存为 'training_metrics_v2.png'")
    plt.show()

# ==========================================
# 5. 主训练循环
# ==========================================
def main():
    train_loader, test_loader, num_classes = get_dataloaders(min_faces=5, batch_size=64)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")
    
    model = LFWFeatureExtractor(embedding_dim=512).to(device)
    arcface = ArcMarginProduct(in_features=512, out_features=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(arcface.parameters())
    
    # 【修改点 3】：加入 weight_decay=1e-4 进行权重衰减 (L2正则化)
    optimizer = optim.Adam(trainable_params, lr=0.0005, weight_decay=1e-4)
    
    epochs = 50  
    print(f"\n[2/3] 开始双层解冻训练 (共 {epochs} Epochs)...")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 【修改点 4】：初始化最优准确率，用于早停保存
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- 训练阶段 ---
        model.train()
        arcface.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(images)
            outputs = arcface(features, labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # --- 验证阶段 ---
        model.eval()
        arcface.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = model(images)
                outputs = arcface(features, labels)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        time_elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {time_elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 【修改点 4 继续】：见好就收，只保存巅峰模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 覆盖保存最优模型 (剔除分类头)
            torch.save(model.backbone.state_dict(), 'my_lfw_resnet18_weights_v2.pth')
            print(f"    🌟 验证集准确率提升至 {best_val_acc:.2f}%，已保存当前最优权重！")

    print(f"\n训练完成！最佳验证集准确率为: {best_val_acc:.2f}%")
    print("-> 最终巅峰权重已保存为 'my_lfw_resnet18_weights_v2.pth'")

    plot_metrics(history)

if __name__ == "__main__":
    main()