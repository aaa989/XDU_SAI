import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time
import seaborn as sns
import warnings
import pandas as pd

# 过滤警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def dimensionality_reduction_experiment():
    """维数约简实验主函数"""

    # 实验1：高维合成数据分类
    synthetic_results = synthetic_experiment()

    # 实验2：手写数字分类
    digits_results = digits_experiment()

    # 实验3：可视化分析
    visualization_analysis()

    # 实验4：性能比较分析
    performance_comparison()

    return synthetic_results, digits_results


def check_data_quality(X, y):
    """检查数据质量"""
    print("\n数据质量检查:")
    print(f"类别分布: {np.bincount(y)}")
    print(f"数据范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"数据均值: {np.mean(X, axis=0)[:3]}...")  # 显示前3个特征
    print(f"数据标准差: {np.std(X, axis=0)[:3]}...")


def synthetic_experiment():
    """高维合成数据实验"""

    print("生成高维合成数据...")
    # 生成更好的高维数据
    X, y = make_classification(
        n_samples=1500,
        n_features=100,  # 高维度
        n_informative=30,  # 30个信息特征
        n_redundant=50,  # 50个冗余特征
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,  # 更复杂的结构
        random_state=42,
        flip_y=0.02,  # 少量噪声
        class_sep=1.2  # 较好的类别分离
    )

    check_data_quality(X, y)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 应用PCA
    pca = PCA(n_components=0.95)  # 保留95%方差
    start_time = time.time()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_time = time.time() - start_time

    print(f"\nPCA降维结果:")
    print(f"原始维度: {X_train.shape[1]}")
    print(f"降维后维度: {X_train_pca.shape[1]}")
    print(f"保留方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")

    # 使用对维度敏感的算法
    algorithms = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }

    results = {}

    for algo_name, model in algorithms.items():

        # 原始数据性能
        start_time = time.time()
        model_original = model.__class__(**model.get_params())
        model_original.fit(X_train, y_train)
        train_time_original = time.time() - start_time

        start_time = time.time()
        y_pred_original = model_original.predict(X_test)
        predict_time_original = time.time() - start_time

        accuracy_original = accuracy_score(y_test, y_pred_original)
        total_time_original = train_time_original + predict_time_original

        # PCA降维后性能
        start_time = time.time()
        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        train_time_pca = time.time() - start_time

        start_time = time.time()
        y_pred_pca = model_pca.predict(X_test_pca)
        predict_time_pca = time.time() - start_time

        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        total_time_pca = train_time_pca + predict_time_pca + pca_time  # 包含PCA时间

        print(f"原始数据 - 准确率: {accuracy_original:.4f}, 总时间: {total_time_original:.3f}s")
        print(f"PCA降维 - 准确率: {accuracy_pca:.4f}, 总时间: {total_time_pca:.3f}s")

        # 计算改善程度
        if total_time_original > 0.001:
            time_reduction = (total_time_original - total_time_pca) / total_time_original * 100
            print(f"总时间变化: {time_reduction:+.1f}%")

        accuracy_change = (accuracy_pca - accuracy_original) * 100
        print(f"准确率变化: {accuracy_change:+.2f}%")

        results[algo_name] = {
            'original_accuracy': accuracy_original,
            'pca_accuracy': accuracy_pca,
            'original_time': total_time_original,
            'pca_time': total_time_pca,
            'original_dims': X_train.shape[1],
            'pca_dims': X_train_pca.shape[1]
        }

    return results


def digits_experiment():
    """手写数字数据集实验"""

    # 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target

    print("手写数字数据集信息:")
    print(f"原始数据维度: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 应用PCA
    pca = PCA(n_components=0.95)  # 保留95%方差
    start_time = time.time()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_time = time.time() - start_time

    print(f"\nPCA降维结果:")
    print(f"原始维度: {X_train.shape[1]}")
    print(f"降维后维度: {X_train_pca.shape[1]}")
    print(f"保留方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"主成分数量: {pca.n_components_}")
    print(f"PCA变换时间: {pca_time:.3f}s")

    # 使用对维度敏感的算法
    algorithms = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }

    results = {}

    for algo_name, model in algorithms.items():
        print(f"\n--- {algo_name} 性能比较 ---")

        # 原始数据性能
        start_time = time.time()
        model_original = model.__class__(**model.get_params())
        model_original.fit(X_train, y_train)
        train_time_original = time.time() - start_time

        start_time = time.time()
        y_pred_original = model_original.predict(X_test)
        predict_time_original = time.time() - start_time

        accuracy_original = accuracy_score(y_test, y_pred_original)
        total_time_original = train_time_original + predict_time_original

        # PCA降维后性能
        start_time = time.time()
        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        train_time_pca = time.time() - start_time

        start_time = time.time()
        y_pred_pca = model_pca.predict(X_test_pca)
        predict_time_pca = time.time() - start_time

        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        total_time_pca = train_time_pca + predict_time_pca + pca_time

        print(f"原始数据 - 准确率: {accuracy_original:.4f}, 总时间: {total_time_original:.3f}s")
        print(f"PCA降维 - 准确率: {accuracy_pca:.4f}, 总时间: {total_time_pca:.3f}s")

        # 计算改善程度
        if total_time_original > 0.001:
            time_reduction = (total_time_original - total_time_pca) / total_time_original * 100
            print(f"总时间变化: {time_reduction:+.1f}%")

        accuracy_change = (accuracy_pca - accuracy_original) * 100
        print(f"准确率变化: {accuracy_change:+.2f}%")

        # 显示详细分类报告
        if algo_name == 'KNN (k=5)':
            print(f"\n{algo_name} 分类报告 (PCA降维后):")
            print(classification_report(y_test, y_pred_pca))

        results[algo_name] = {
            'original_accuracy': accuracy_original,
            'pca_accuracy': accuracy_pca,
            'original_time': total_time_original,
            'pca_time': total_time_pca,
            'original_dims': X_train.shape[1],
            'pca_dims': X_train_pca.shape[1]
        }

    return results


def visualization_analysis():
    """可视化分析实验"""

    # 生成用于可视化的高维数据
    X, y = make_classification(
        n_samples=800,
        n_features=80,
        n_informative=20,
        n_redundant=40,
        n_classes=3,
        random_state=42,
        class_sep=1.5
    )

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 应用t-SNE
    print("正在进行t-SNE降维可视化...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 特征相关性热力图（只显示前15个特征）
    n_features_show = min(15, X_scaled.shape[1])
    corr_matrix = np.corrcoef(X_scaled[:, :n_features_show].T)

    im = axes[0, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title(f'前{n_features_show}个特征的相关性热力图', fontsize=12)
    axes[0, 0].set_xlabel('特征索引')
    axes[0, 0].set_ylabel('特征索引')
    plt.colorbar(im, ax=axes[0, 0])

    # 添加相关性数值
    for i in range(n_features_show):
        for j in range(n_features_show):
            axes[0, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center', fontsize=8)

    # 2. PCA主成分方差解释
    pca_full = PCA().fit(X_scaled)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

    axes[0, 1].plot(range(1, len(explained_variance) + 1), explained_variance,
                    'b-', linewidth=2, label='累积解释方差')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%方差线')
    axes[0, 1].set_xlabel('主成分数量')
    axes[0, 1].set_ylabel('累积解释方差比例')
    axes[0, 1].set_title('PCA方差解释比例', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # 标记95%方差对应的主成分数
    n_components_95 = np.argmax(explained_variance >= 0.95) + 1
    axes[0, 1].axvline(x=n_components_95, color='g', linestyle=':', alpha=0.7,
                       label=f'95%方差: {n_components_95}个主成分')
    axes[0, 1].legend()

    # 3. PCA降维可视化
    scatter_pca = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                                     cmap='viridis', alpha=0.7, s=40)
    axes[1, 0].set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0] * 100:.1f}%方差)')
    axes[1, 0].set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1] * 100:.1f}%方差)')
    axes[1, 0].set_title('PCA降维可视化 (2D)', fontsize=12)
    cbar = plt.colorbar(scatter_pca, ax=axes[1, 0])
    cbar.set_label('类别')

    # 4. t-SNE降维可视化
    scatter_tsne = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                                      cmap='viridis', alpha=0.7, s=40)
    axes[1, 1].set_xlabel('t-SNE维度1')
    axes[1, 1].set_ylabel('t-SNE维度2')
    axes[1, 1].set_title('t-SNE降维可视化 (2D)', fontsize=12)
    cbar = plt.colorbar(scatter_tsne, ax=axes[1, 1])
    cbar.set_label('类别')

    plt.tight_layout()
    plt.show()

    # 维度灾难演示
    demonstrate_curse_of_dimensionality()


def demonstrate_curse_of_dimensionality():
    """维度灾难演示"""

    print("\n维度灾难演示:")
    print("-" * 40)

    # 在不同维度下生成随机数据并计算距离
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]
    n_samples = 300

    distance_ratios = []

    for d in dimensions:
        # 生成随机数据
        np.random.seed(42)
        data = np.random.randn(n_samples, d)

        # 计算最近邻距离
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, _ = nbrs.kneighbors(data)
        nearest_dist = np.mean(distances[:, 1])  # 最近邻距离

        # 计算理论平均距离
        mean_dist = np.sqrt(2 * d)  # 对于标准正态分布

        ratio = nearest_dist / mean_dist
        distance_ratios.append(ratio)

        print(f"维度 {d:3d}: 最近邻/平均距离比 = {ratio:.4f}")

    # 绘制维度灾难图
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, distance_ratios, 'bo-', linewidth=2, markersize=6)
    plt.xscale('log')
    plt.xlabel('数据维度 (对数尺度)')
    plt.ylabel('最近邻距离 / 平均距离')
    plt.title('维度灾难演示: 高维空间中距离度量失效')
    plt.grid(True, alpha=0.3)

    # 添加说明文本
    plt.text(10, 0.7, '维度增加 → 距离比趋近1\n所有数据点变得"相似"\n基于距离的算法失效',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=10)

    plt.tight_layout()
    plt.show()


def performance_comparison():
    """性能比较实验 - 展示不同维度下的性能变化"""

    # 生成高维数据
    X, y = make_classification(
        n_samples=1200,
        n_features=120,
        n_informative=25,
        n_redundant=60,
        n_classes=3,
        random_state=42,
        class_sep=1.3
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 测试不同的降维程度
    variance_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    accuracies = []
    training_times = []
    dimensions_after_pca = []

    print("\n不同降维程度下的KNN性能比较:")
    print("-" * 70)
    print("保留方差\t维度\t准确率\t训练时间(s)\t预测时间(s)\t总时间(s)")
    print("-" * 70)

    # 使用KNN（对维度敏感）
    model = KNeighborsClassifier(n_neighbors=5)

    # 先测试原始数据（不降维）
    start_time = time.time()
    model_original = model.__class__(**model.get_params())
    model_original.fit(X_train, y_train)
    training_time_original = time.time() - start_time

    start_time = time.time()
    y_pred_original = model_original.predict(X_test)
    prediction_time_original = time.time() - start_time

    accuracy_original = accuracy_score(y_test, y_pred_original)
    total_time_original = training_time_original + prediction_time_original

    print(
        f"原始数据\t{X_train.shape[1]:3d}\t{accuracy_original:.4f}\t{training_time_original:.3f}\t\t{prediction_time_original:.3f}\t\t{total_time_original:.3f}")

    for variance in variance_levels:
        # 应用PCA
        pca = PCA(n_components=variance)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        dimensions_after_pca.append(X_train_pca.shape[1])

        # 训练模型并计时
        start_time = time.time()
        model_pca = model.__class__(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        training_time = time.time() - start_time

        # 预测并计时
        start_time = time.time()
        y_pred = model_pca.predict(X_test_pca)
        prediction_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        total_time = training_time + prediction_time

        accuracies.append(accuracy)
        training_times.append(total_time)

        print(
            f"{variance:.2f}\t\t{X_train_pca.shape[1]:3d}\t{accuracy:.4f}\t{training_time:.3f}\t\t{prediction_time:.3f}\t\t{total_time:.3f}")

    # 将所有结果合并（包括原始数据）
    all_dimensions = [X_train.shape[1]] + dimensions_after_pca
    all_accuracies = [accuracy_original] + accuracies
    all_times = [total_time_original] + training_times

    # 绘制性能比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 准确率vs维度
    ax1.plot(all_dimensions, all_accuracies, 'go-', linewidth=2, markersize=6)
    ax1.set_xlabel('数据维度')
    ax1.set_ylabel('KNN分类准确率')
    ax1.set_title('KNN准确率 vs 数据维度', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 标记最优准确率点
    best_idx = np.argmax(all_accuracies)
    ax1.plot(all_dimensions[best_idx], all_accuracies[best_idx], 'ro', markersize=10,
             label=f'最佳: 维度={all_dimensions[best_idx]}, 准确率={all_accuracies[best_idx]:.3f}')
    ax1.legend()

    # 训练时间vs维度
    ax2.plot(all_dimensions, all_times, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('数据维度')
    ax2.set_ylabel('总时间 (秒)')
    ax2.set_title('KNN总时间 vs 数据维度', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 找到最佳平衡点（准确率与时间的权衡）
    normalized_accuracy = (np.array(all_accuracies) - min(all_accuracies)) / (max(all_accuracies) - min(all_accuracies))
    normalized_time = 1 - (np.array(all_times) - min(all_times)) / (max(all_times) - min(all_times))
    combined_score = normalized_accuracy + normalized_time

    best_balance_idx = np.argmax(combined_score)

    print(f"\n最佳平衡点分析:")
    print(f"最佳平衡维度: {all_dimensions[best_balance_idx]}")
    print(f"对应准确率: {all_accuracies[best_balance_idx]:.4f}")
    print(f"对应总时间: {all_times[best_balance_idx]:.3f}s")
    print(f"相比原始数据:")
    print(f"  准确率变化: {(all_accuracies[best_balance_idx] - accuracy_original) * 100:+.2f}%")
    print(f"  时间变化: {(all_times[best_balance_idx] - total_time_original) * 100 / total_time_original:+.1f}%")


def main():
    """主函数"""
    try:
        # 运行主实验
        synthetic_results, digits_results = dimensionality_reduction_experiment()

    except Exception as e:
        print(f"实验运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()