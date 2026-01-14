import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# 加载 Iris 数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 加载 Sonar 数据集（需确保数据文件在指定路径，这里假设从UCI获取后保存为sonar.csv）
# 数据格式为：前60列为特征，最后一列为标签（'R'表示岩石，'M'表示水雷，转换为0和1）
sonar_data = np.genfromtxt('sonar.csv', delimiter=',', dtype=None, encoding=None,skip_header=1)
X_sonar = np.array([list(row)[:-1] for row in sonar_data], dtype=float)
y_sonar = np.array([0 if row[-1] == 'R' else 1 for row in sonar_data])

# 数据预处理：标准化
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

scaler_sonar = StandardScaler()
X_sonar_scaled = scaler_sonar.fit_transform(X_sonar)

# 定义 k 折交叉验证的折数
k = 100

# 对 Iris 数据集进行 k 折交叉验证
kf_iris = KFold(n_splits=k, shuffle=True, random_state=42)
lda_iris = LinearDiscriminantAnalysis()
iris_scores = cross_val_score(lda_iris, X_iris_scaled, y_iris, cv=kf_iris)

# 对 Sonar 数据集进行 k 折交叉验证
kf_sonar = KFold(n_splits=k, shuffle=True, random_state=42)
lda_sonar = LinearDiscriminantAnalysis()
sonar_scores = cross_val_score(lda_sonar, X_sonar_scaled, y_sonar, cv=kf_sonar)

# 计算平均准确率
iris_avg_accuracy = np.mean(iris_scores)
sonar_avg_accuracy = np.mean(sonar_scores)

# 打印结果
print(f"Iris 数据集 {k}-折交叉验证的平均准确率: {iris_avg_accuracy:.4f}")
print(f"Sonar 数据集 {k}-折交叉验证的平均准确率: {sonar_avg_accuracy:.4f}")

# 绘制不同数据集上的准确率对比条形图
datasets = ['Iris', 'Sonar']
accuracies = [iris_avg_accuracy, sonar_avg_accuracy]

plt.bar(datasets, accuracies, color=['skyblue', 'lightgreen'])
plt.title(f'{k}-Fold Cross Validation Accuracy for LDA')
plt.xlabel('Dataset')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
plt.show()