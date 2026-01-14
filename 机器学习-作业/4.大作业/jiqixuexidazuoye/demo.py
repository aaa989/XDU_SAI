import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from scipy.signal import medfilt, welch
import os

# -------------------------- 1. 数据集配置与加载 --------------------------
DATA_PATH = "D:/app/基本组件/Desktop/jiqixuexidazuoye/data/"

FAULT_CLASSES = {
    "Normal": 0,
    "Ball_0.007": 1,
    "InnerRace_0.007": 2,
    "OuterRace_0.007_6": 3,
    "Ball_0.014": 4,
    "InnerRace_0.014": 5,
    "OuterRace_0.014_6": 6,
    "Ball_0.021": 7,
    "InnerRace_0.021": 8,
    "OuterRace_0.021_6": 9
}



def load_mat_file(file_path, signal_type="DE"):

    mat_data = sio.loadmat(file_path)

    signal_keys = [key for key in mat_data.keys() if signal_type.upper() in key and ('time' in key or key.islower())]
    if not signal_keys:
        signal_keys = [key for key in mat_data.keys() if signal_type.upper() in key]
    signal_key = signal_keys[0] if signal_keys else None

    signal = mat_data[signal_key].flatten()
    return signal

def build_dataset(data_path):
    X, y = [], []
    signal_length = 1024
    step = 512

    file_mapping = {
        "Normal": ["97.mat", "98.mat", "99.mat", "100.mat"],
        "Ball_0.007": ["118.mat", "119.mat", "120.mat", "121.mat"],
        "InnerRace_0.007": ["105.mat", "106.mat", "107.mat", "108.mat"],
        "OuterRace_0.007_6": ["130.mat", "131.mat", "132.mat", "133.mat"],
        "Ball_0.014": ["185.mat", "186.mat", "187.mat", "188.mat"],
        "InnerRace_0.014": ["169.mat", "170.mat", "171.mat", "172.mat"],
        "OuterRace_0.014_6": ["197.mat", "198.mat", "199.mat", "200.mat"],
        "Ball_0.021": ["222.mat", "223.mat", "224.mat", "225.mat"],
        "InnerRace_0.021": ["209.mat", "210.mat", "211.mat", "212.mat"],
        "OuterRace_0.021_6": ["234.mat", "235.mat", "236.mat", "237.mat"]
    }

    for fault_type, file_names in file_mapping.items():
        label = FAULT_CLASSES[fault_type]
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            if not os.path.exists(file_path):
                print(f"警告：文件{file_path}不存在，跳过")
                continue
            try:
                signal = load_mat_file(file_path, signal_type="DE")
            except Exception as e:
                print(f"警告：加载文件{file_path}失败，{str(e)}，跳过")
                continue
            max_start = len(signal) - signal_length
            if max_start <= 0:
                print(f"警告：文件{file_path}的信号长度不足，跳过")
                continue
            for i in range(0, max_start, step):
                signal_segment = signal[i:i + signal_length]
                X.append(signal_segment)
                y.append(label)

    return np.array(X), np.array(y)


# 加载并划分数据集（7:2:1）
X, y = build_dataset(DATA_PATH)
if len(X) == 0:
    raise ValueError("未加载到任何数据")
print(f"总样本数：{len(X)}")

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=2 / 9, random_state=42,
                                                  stratify=y_train_val)

print(f"训练集：{X_train.shape}, 验证集：{X_val.shape}, 测试集：{X_test.shape}")


# -------------------------- 2. 数据预处理 --------------------------
def preprocess_signals(X):

    X_processed = []
    scaler = StandardScaler()
    for signal in X:

        denoised = medfilt(signal, kernel_size=5)
        normalized = scaler.fit_transform(denoised.reshape(-1, 1)).flatten()
        X_processed.append(normalized)
    return np.array(X_processed)

X_train = preprocess_signals(X_train)
X_val = preprocess_signals(X_val)
X_test = preprocess_signals(X_test)


# -------------------------- 3. 特征提取 --------------------------
def extract_expert_features(signals):

    features = []
    for signal in signals:
        # ---------------------- 时域特征（10维）----------------------
        time_mean = np.mean(signal)
        time_var = np.var(signal)
        time_std = np.std(signal)
        time_kurt = kurtosis(signal)  # 峭度
        time_skew = skew(signal)  # 偏度
        time_peak = np.max(np.abs(signal))  # 峰值
        time_peak2peak = np.ptp(signal)  # 峰峰值
        time_rms = np.sqrt(np.mean(signal ** 2))  # 均方根
        time_impulse = time_peak / time_rms  # 脉冲因子
        time_clearance = time_peak / np.power(np.mean(np.abs(signal)), 1 / 2)  # 裕度因子

        # ---------------------- 频域特征（10维）----------------------
        # 傅里叶变换
        freq = fft(signal)
        freq_amp = np.abs(freq)[:len(signal) // 2]
        freq_axis = np.fft.fftfreq(len(signal), d=1 / 12000)[:len(signal) // 2]  # 12kHz采样率

        # 基础频域特征
        freq_mean = np.mean(freq_amp)
        freq_var = np.var(freq_amp)
        freq_peak = np.max(freq_amp)
        freq_rms = np.sqrt(np.mean(freq_amp ** 2))

        # 功率谱密度（PSD）特征
        f_psd, psd = welch(signal, fs=12000, nperseg=512)
        psd_mean = np.mean(psd)
        psd_peak = np.max(psd)
        psd_center = np.sum(f_psd * psd) / np.sum(psd)  # 重心频率
        psd_rms = np.sqrt(np.sum(psd ** 2) / len(psd))  # PSD均方根

        # 谱峭度和谱偏度
        freq_kurt = kurtosis(freq_amp)
        freq_skew = skew(freq_amp)

        # 拼接所有特征（20维）
        feature = [
            time_mean, time_var, time_std, time_kurt, time_skew,
            time_peak, time_peak2peak, time_rms, time_impulse, time_clearance,
            freq_mean, freq_var, freq_peak, freq_rms, freq_kurt,
            freq_skew, psd_mean, psd_peak, psd_center, psd_rms
        ]
        features.append(feature)
    return np.array(features)


# 提取专家特征
X_train_expert = extract_expert_features(X_train)
X_val_expert = extract_expert_features(X_val)
X_test_expert = extract_expert_features(X_test)


# -------------------------- 4. CNN深层特征提取 --------------------------
class SignalCNN(nn.Module):

    def __init__(self):
        super(SignalCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 128, 128)
    def forward(self, x):

        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 数据集类
class BearingDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.reshape(-1, 1, 1024)
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return torch.tensor(self.signals[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_cnn_feature_extractor():

    train_dataset = BearingDataset(X_train, y_train)
    val_dataset = BearingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    epochs = 50
    best_val_acc = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"训练CNN设备：{device}）")
    for epoch in range(epochs):

        model.train()
        train_loss, train_correct = 0.0, 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * signals.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        train_loss_avg = train_loss / len(train_dataset)
        val_loss_avg = val_loss / len(val_dataset)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn.pth")

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("CNN Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc", linewidth=2)
    plt.plot(val_accs, label="Val Acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("CNN Accuracy Curve")
    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    model.load_state_dict(torch.load("best_cnn.pth"))
    model.eval()

    def extract_cnn_feat(signals):
        dataset = BearingDataset(signals, np.zeros(len(signals)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        cnn_feats = []
        with torch.no_grad():
            for signals, _ in loader:
                signals = signals.to(device)
                feats = model(signals)
                cnn_feats.extend(feats.cpu().numpy())
        return np.array(cnn_feats)

    X_train_cnn = extract_cnn_feat(X_train)
    X_val_cnn = extract_cnn_feat(X_val)
    X_test_cnn = extract_cnn_feat(X_test)

    return X_train_cnn, X_val_cnn, X_test_cnn

X_train_cnn, X_val_cnn, X_test_cnn = train_cnn_feature_extractor()

X_train_fused = np.hstack([X_train_expert, X_train_cnn])
X_val_fused = np.hstack([X_val_expert, X_val_cnn])
X_test_fused = np.hstack([X_test_expert, X_test_cnn])

scaler_fused = StandardScaler()
X_train_fused = scaler_fused.fit_transform(X_train_fused)
X_val_fused = scaler_fused.transform(X_val_fused)
X_test_fused = scaler_fused.transform(X_test_fused)

print(f"融合后特征维度：{X_train_fused.shape[1]}")

# -------------------------- 5. SVM分类器训练与优化 --------------------------

param_grid = {
    "C": [1, 10, 100, 200],
    "gamma": [0.001, 0.01, 0.1, 1],
    "kernel": ["rbf"]
}

svm = SVC(random_state=42, probability=True)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train_fused, y_train)

best_svm = grid_search.best_estimator_
print(f"最佳SVM参数：{grid_search.best_params_}")
print(f"训练集交叉验证准确率：{grid_search.best_score_:.4f}")

y_val_pred = best_svm.predict(X_val_fused)
y_test_pred = best_svm.predict(X_test_fused)

val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="weighted")

print(f"\n验证集准确率：{val_acc:.4f}")
print(f"测试集准确率：{test_acc:.4f}")
print(f"测试集加权F1分数：{test_f1:.4f}")

# -------------------------- 6. 结果可视化 --------------------------
cm = confusion_matrix(y_test, y_test_pred)
class_names = list(FAULT_CLASSES.keys())

plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap=plt.cm.Blues, xticks_rotation=45, colorbar=True
)
plt.title("Bearing Fault Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

experiment_results = f"""
性能指标：
   - 验证集准确率：{val_acc:.4f}
   - 测试集准确率：{test_acc:.4f}
   - 测试集加权F1：{test_f1:.4f}
"""

with open("experiment_results.txt", "w", encoding="utf-8") as f:
    f.write(experiment_results)

print("\n" + "=" * 50)
print(experiment_results)