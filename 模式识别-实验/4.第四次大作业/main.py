import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --------------------- 1. 加载并预处理数据集 ---------------------
# 加载Iris数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
# 加载Sonar数据集
sonar = fetch_openml(name='sonar', version=1)
X_sonar, y_sonar = sonar.data, sonar.target

# 数据标准化
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
scaler_sonar = StandardScaler()
X_sonar_scaled = scaler_sonar.fit_transform(X_sonar)

# 划分训练集和测试集（8:2）
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.2, random_state=42
)
X_sonar_train, X_sonar_test, y_sonar_train, y_sonar_test = train_test_split(
    X_sonar_scaled, y_sonar, test_size=0.2, random_state=42
)

# --------------------- 2. 定义SVM模型并测试不同核函数 ---------------------
kernels = ['linear', 'poly', 'rbf']  # 三种核函数

# ===== Iris数据集（多分类）测试 =====
print("===== Iris数据集 SVM 分类结果 =====")
for kernel in kernels:
    svm_iris = SVC(kernel=kernel, random_state=42)
    svm_iris.fit(X_iris_train, y_iris_train)
    y_iris_pred = svm_iris.predict(X_iris_test)
    acc = accuracy_score(y_iris_test, y_iris_pred)
    print(f"核函数: {kernel}, 准确率: {acc:.4f}")
    print(classification_report(y_iris_test, y_iris_pred, target_names=iris.target_names))
    print("-" * 50)

# ===== Sonar数据集（二分类）测试 =====
print("===== Sonar数据集 SVM 分类结果 =====")
for kernel in kernels:
    svm_sonar = SVC(kernel=kernel, random_state=42)
    svm_sonar.fit(X_sonar_train, y_sonar_train)
    y_sonar_pred = svm_sonar.predict(X_sonar_test)
    acc = accuracy_score(y_sonar_test, y_sonar_pred)
    print(f"核函数: {kernel}, 准确率: {acc:.4f}")
    print(classification_report(y_sonar_test, y_sonar_pred, target_names=['Rock', 'Mine']))
    print("-" * 50)
