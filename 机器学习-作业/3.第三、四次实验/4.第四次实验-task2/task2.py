import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def train_with_pca_stacking():
    start_time = time.time()
    print(">>> [1/6] 正在读取数据...")
    train_df = pd.read_csv('train.csv', header=0)
    test_df = pd.read_csv('test.csv', header=0)

    # 1. 提取数据
    X = train_df.iloc[:, 0:2048].values
    y = train_df.iloc[:, 2048].values
    X_test_submit = test_df.iloc[:, 0:2048].values

    # 2. 标准化 (PCA前必须步骤)
    print(">>> [2/6] 正在进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_submit)

    # 3. 高保真 PCA 降维
    # 策略：为了冲击95%准确率，我们设置 n_components=0.99
    # 这意味着保留99%的原始方差，只丢弃1%的噪音
    print(">>> [3/6] 正在进行高保真 PCA 降维 (保留99%信息)...")
    pca = PCA(n_components=0.99, random_state=42)
    
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"   维度变化: 2048 -> {X_pca.shape[1]}")
    print("   (维度降低有助于 SVM 和 神经网络 更好更快地收敛)")

    # 4. 构建 Stacking 模型
    print(">>> [4/6] 构建 Stacking 融合模型...")

    # 基模型 1: SVM (降维后的王者)
    svm_clf = SVC(
        kernel='rbf', 
        C=20,               # 适当降低一点C，防止在压缩特征上过拟合
        gamma='scale', 
        probability=True, 
        cache_size=1000,
        random_state=42
    )

    # 基模型 2: LightGBM (速度快，精度高)
    lgbm_clf = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 基模型 3: MLP (神经网络)
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(512, 256), # 输入维度变小了，隐藏层也可以稍微减小
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=800,
        early_stopping=True,
        random_state=42
    )

    # 堆叠分类器
    stacking_model = StackingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('lgbm', lgbm_clf),
            ('mlp', mlp_clf)
        ],
        final_estimator=LogisticRegression(C=10),
        n_jobs=-1,
        cv=5
    )

    # 5. 训练
    print(">>> [5/6] 开始训练融合模型 (PCA加速版)...")
    
    # 交叉验证检查精度
    scores = cross_val_score(stacking_model, X_pca, y, cv=5, scoring='accuracy')
    print(f"   === 交叉验证结果 ===")
    print(f"   平均准确率: {scores.mean()*100:.2f}%")
    
    # 正式训练
    stacking_model.fit(X_pca, y)

    # 6. 预测与保存
    print(">>> [6/6] 正在生成结果...")
    test_preds = stacking_model.predict(X_test_pca)

    label_col_name = test_df.columns[2048]
    test_df[label_col_name] = test_preds

    output_file = 'predicted_test_pca_stacking.csv'
    test_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"全部完成！耗时: {end_time - start_time:.1f} 秒")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    train_with_pca_stacking()