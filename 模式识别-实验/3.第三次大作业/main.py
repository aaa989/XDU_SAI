import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from PIL import Image
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]  
plt.rcParams["axes.unicode_minus"] = False 

# --------------------- 工具函数---------------------
def preprocess_image_data(X, n_components=100):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"图像数据降维完成：{X.shape[1]}维 → {n_components}维（保留信息量：{sum(pca.explained_variance_ratio_):.2f}）")
    return X_pca

def sample_dataset(X, y, samples_per_class=50):
    unique_classes = np.unique(y)
    X_sampled, y_sampled = [], []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        sample_indices = np.random.choice(cls_indices, min(samples_per_class, len(cls_indices)), replace=False)
        X_sampled.extend(X[sample_indices])
        y_sampled.extend(y[sample_indices])
    return np.array(X_sampled), np.array(y_sampled)

def knn_comparison(dataset1, dataset2, k_range=range(1, 21), cv=10):
    X1, y1, name1 = dataset1
    X2, y2, name2 = dataset2

    errors1, errors2 = [], []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        acc1 = cross_val_score(knn, X1, y1, cv=cv, scoring='accuracy').mean()
        errors1.append(1 - acc1)
        acc2 = cross_val_score(knn, X2, y2, cv=cv, scoring='accuracy').mean()
        errors2.append(1 - acc2)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, errors1, 'o-', color='blue', label=name1)
    plt.plot(k_range, errors2, 's-', color='red', label=name2)
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('错误率', fontsize=12)
    plt.title(f'KNN在{name1}与{name2}上的错误率对比', fontsize=14)
    plt.xticks(k_range)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.show()

    optimal_k1 = k_range[np.argmin(errors1)]
    optimal_k2 = k_range[np.argmin(errors2)]
    print(f"\n{name1}最优K值: {optimal_k1}, 最小错误率: {min(errors1):.4f}")
    print(f"{name2}最优K值: {optimal_k2}, 最小错误率: {min(errors2):.4f}\n")

# --------------------- 数据集加载函数 ---------------------
def load_iris_sonar():
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    sonar_path = "sonar.csv"
    sonar_data = np.genfromtxt(sonar_path, delimiter=',', skip_header=1, encoding='utf-8', dtype=object)
    X_sonar = sonar_data[:, :-1].astype(float)
    y_sonar = np.array([0 if label == 'R' else 1 for label in sonar_data[:, -1]])

    scaler = StandardScaler()
    X_iris = scaler.fit_transform(X_iris)
    X_sonar = scaler.fit_transform(X_sonar)
    return (X_iris, y_iris, "Iris"), (X_sonar, y_sonar, "Sonar")

def load_mnist_cifar10(n_samples=1000, n_components=100):
    # --------------------- MNIST加载部分 ---------------------
    import os
    mnist_path = "./data/openml/mnist.npz"  # 手动下载的文件路径
    if os.path.exists(mnist_path):
        # 直接读取本地MNIST文件
        with np.load(mnist_path, allow_pickle=True) as f:
            X_mnist, y_mnist = f['x_train'], f['y_train']
        X_mnist = X_mnist.reshape(-1, 784)  # 展平为784维特征（28x28）
        y_mnist = y_mnist.astype(int)
        X_mnist, y_mnist = X_mnist[:n_samples], y_mnist[:n_samples]

    # --------------------- CIFAR-10部分 ---------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().flatten())
    ])
    cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    X_cifar = np.array([cifar[i][0] for i in range(n_samples)])
    y_cifar = np.array([cifar[i][1] for i in range(n_samples)])

    X_mnist = preprocess_image_data(X_mnist, n_components)
    X_cifar = preprocess_image_data(X_cifar, n_components)
    return (X_mnist, y_mnist, "MNIST"), (X_cifar, y_cifar, "CIFAR-10")

def load_ucm_nwpu(ucm_path, nwpu_path, samples_per_class=50, n_components=100):
    def load_remote_sensing(path):
        X, y = [], []
        classes = os.listdir(path)
        for cls_idx, cls_name in enumerate(classes):
            cls_path = os.path.join(path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    img = Image.open(img_path).resize((128, 128))
                    img_arr = np.array(img).flatten()
                    X.append(img_arr)
                    y.append(cls_idx)
        X, y = np.array(X), np.array(y)
        X_sampled, y_sampled = sample_dataset(X, y, samples_per_class)
        return X_sampled, y_sampled

    X_ucm, y_ucm = load_remote_sensing(ucm_path)
    X_nwpu, y_nwpu = load_remote_sensing(nwpu_path)

    X_ucm = preprocess_image_data(X_ucm, n_components)
    X_nwpu = preprocess_image_data(X_nwpu, n_components)
    return (X_ucm, y_ucm, "UCM"), (X_nwpu, y_nwpu, "NWPU")


# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    print("===== 第一组对比：Iris（植物） vs Sonar（声呐） =====")
    dataset_iris, dataset_sonar = load_iris_sonar()
    knn_comparison(dataset_iris, dataset_sonar)

    print("===== 第二组对比：MNIST（手写数字） vs CIFAR-10（彩色图像） =====")
    dataset_mnist, dataset_cifar = load_mnist_cifar10(n_samples=1000, n_components=100)
    knn_comparison(dataset_mnist, dataset_cifar)

    print("===== 第三组对比：UCM（遥感） vs NWPU（遥感） =====")
    ucm_path = "./UCMerced_LandUse/Images"
    nwpu_path = "./NWPU-RESISC45"
    if os.path.exists(ucm_path) and os.path.exists(nwpu_path):
        dataset_ucm, dataset_nwpu = load_ucm_nwpu(
            ucm_path=ucm_path,
            nwpu_path=nwpu_path,
            samples_per_class=50,
            n_components=100
        )
        knn_comparison(dataset_ucm, dataset_nwpu)
