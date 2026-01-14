import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random
import os
import copy

# 1. 锁定种子
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 运行设备: {device}")

# ---------------------------------------------------------
# 2. 模型：Small DenseNet (针对 16x16 优化的稠密网络)
# ---------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], 1) # 稠密连接：拼接特征

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2) # 下采样

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class SmallDenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=12):
        super(SmallDenseNet, self).__init__()
        # 初始卷积：不降采样
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False)
        
        # 16x16 -> 16x16
        self.block1 = DenseBlock(24, growth_rate, n_layers=6)
        in_channels = 24 + 6 * growth_rate
        self.trans1 = Transition(in_channels, in_channels // 2)
        
        # 8x8 -> 8x8
        in_channels = in_channels // 2
        self.block2 = DenseBlock(in_channels, growth_rate, n_layers=12)
        in_channels = in_channels + 12 * growth_rate
        self.trans2 = Transition(in_channels, in_channels // 2)
        
        # 4x4 -> 4x4
        in_channels = in_channels // 2
        self.block3 = DenseBlock(in_channels, growth_rate, n_layers=16)
        in_channels = in_channels + 16 * growth_rate
        
        self.bn_final = nn.BatchNorm2d(in_channels)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ---------------------------------------------------------
# 3. 数据集 (标准化 + 极微弱增强)
# ---------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, X, y=None, mode='train'):
        self.X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 16, 16)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
        self.mode = mode

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]
        if self.mode == 'train' and self.y is not None:
            # 极微弱噪声，防止过拟合
            if random.random() > 0.5:
                img = img + torch.randn_like(img) * 0.01
        
        if self.y is not None:
            return img, self.y[idx]
        return img

# ---------------------------------------------------------
# 4. 训练函数
# ---------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, num_classes, batch_size=128):
    train_ds = MyDataset(X_train, y_train, mode='train')
    val_ds = MyDataset(X_val, y_val, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    model = SmallDenseNet(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    best_weights = None
    
    epochs = 40
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + epoch / len(train_loader))
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            
    return best_weights, best_acc

# ---------------------------------------------------------
# 5. 主程序：伪标签流程
# ---------------------------------------------------------
def main():
    print("📥 读取数据...")
    df_train = pd.read_csv('train.csv', header=0)
    df_test = pd.read_csv('test.csv', header=0)

    X_train_raw = df_train.iloc[:, 0:256].values
    y_train_raw = df_train.iloc[:, 256].values
    X_test_raw = df_test.iloc[:, 0:256].values

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_raw)
    X_test_full = scaler.transform(X_test_raw)

    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)
    
    # -------------------------------------------
    # 第一阶段：训练基础模型 (5折交叉验证)
    # -------------------------------------------
    print("🔥 阶段一：训练基础模型 (5-Fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = []
    
    # 训练 5 个模型
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"  Training Fold {fold+1}...")
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        weights, acc = train_model(X_tr, y_tr, X_val, y_val, num_classes)
        print(f"  Fold {fold+1} Acc: {acc:.4f}")
        
        # 保存模型用于预测伪标签
        model = SmallDenseNet(num_classes).to(device)
        model.load_state_dict(weights)
        model.eval()
        models.append(model)

    # -------------------------------------------
    # 第二阶段：生成伪标签 (Pseudo Labeling)
    # -------------------------------------------
    print("\n🔍 阶段二：生成伪标签...")
    test_ds = MyDataset(X_test_full, mode='test')
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    # 用 5 个模型预测测试集，取平均
    probs_sum = np.zeros((len(X_test_full), num_classes))
    with torch.no_grad():
        for model in models:
            model.eval()
            fold_probs = []
            for imgs in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                fold_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            probs_sum += np.concatenate(fold_probs, axis=0)
            
    avg_probs = probs_sum / 5
    # 获取最高置信度
    max_probs = np.max(avg_probs, axis=1)
    pseudo_labels = np.argmax(avg_probs, axis=1)
    
    # 设定阈值：只有非常确定的样本才加入训练集
    threshold = 0.95
    high_conf_idx = np.where(max_probs > threshold)[0]
    
    print(f"  筛选出 {len(high_conf_idx)} / {len(X_test_full)} 个高置信度样本加入训练集")
    
    # 合并数据
    X_pseudo = X_test_full[high_conf_idx]
    y_pseudo = pseudo_labels[high_conf_idx]
    
    X_final_train = np.concatenate([X_train_full, X_pseudo], axis=0)
    y_final_train = np.concatenate([y_train_full, y_pseudo], axis=0)
    
    print(f"  新训练集大小: {len(X_final_train)}")
    
    # -------------------------------------------
    # 第三阶段：用新数据全量重新训练
    # -------------------------------------------
    print("\n🚀 阶段三：使用伪标签数据重新训练最终模型...")
    # 这里我们不再交叉验证，而是全量训练一个强模型，或者再训练一组ensemble
    # 为了稳妥，我们再训练一个 5-Fold 集成，但是是在扩充后的数据上
    
    final_models = []
    # 重新划分扩充后的数据
    skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=999) # 换个随机种子
    
    for fold, (train_idx, val_idx) in enumerate(skf_final.split(X_final_train, y_final_train)):
        print(f"  Retraining Fold {fold+1} (with Pseudo)...")
        X_tr, X_val = X_final_train[train_idx], X_final_train[val_idx]
        y_tr, y_val = y_final_train[train_idx], y_final_train[val_idx]
        
        # 增加一点训练轮数
        weights, acc = train_model(X_tr, y_tr, X_val, y_val, num_classes)
        print(f"  Fold {fold+1} Acc: {acc:.4f}")
        
        model = SmallDenseNet(num_classes).to(device)
        model.load_state_dict(weights)
        final_models.append(model)
        
    # -------------------------------------------
    # 第四阶段：最终预测
    # -------------------------------------------
    print("\n🏁 最终预测...")
    final_probs_sum = np.zeros((len(X_test_full), num_classes))
    
    with torch.no_grad():
        for model in final_models:
            model.eval()
            fold_probs = []
            for imgs in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                fold_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            final_probs_sum += np.concatenate(fold_probs, axis=0)
            
    final_avg_probs = final_probs_sum / 5
    final_preds = np.argmax(final_avg_probs, axis=1)
    
    final_labels = label_encoder.inverse_transform(final_preds)
    label_name = df_train.columns[256]
    df_test[label_name] = final_labels
    df_test.to_csv('test_result_pseudo_densenet.csv', index=False)
    print("✅ 杀手锏方案执行完毕，结果已保存至 test_result_pseudo_densenet.csv")

if __name__ == "__main__":
    main()