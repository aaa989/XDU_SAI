import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# 1. 加载数据
print("=" * 50)


diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target


# 2. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"标准化后训练集均值: {np.mean(X_train_scaled, axis=0).round(2)}")
print(f"标准化后训练集标准差: {np.std(X_train_scaled, axis=0).round(2)}")

# 4. 训练不同模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Ridge (alpha=10.0)': Ridge(alpha=10.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0)
}

results = {}

# 训练并评估模型
print("\n" + "=" * 30)

for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': model.coef_ if hasattr(model, 'coef_') else None
    }

    print(f"{name:20} | MSE: {mse:8.2f} | R²: {r2:6.3f}")

# 5. 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 5.1 真实值 对比 预测值散点图
best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
best_result = results[best_model_name]

axes[0, 0].scatter(y_test, best_result['predictions'], alpha=0.7, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('真实值')
axes[0, 0].set_ylabel('预测值')
axes[0, 0].set_title(
    f'真实值 vs 预测值 ({best_model_name})\nMSE: {best_result["mse"]:.2f}, R2: {best_result["r2"]:.3f}')
axes[0, 0].grid(True, alpha=0.3)

# 5.2 模型性能比较
model_names = list(results.keys())
mses = [results[name]['mse'] for name in model_names]
r2_scores = [results[name]['r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0, 1].bar(x - width / 2, mses, width, label='MSE', alpha=0.7)
bars2 = axes[0, 1].bar(x + width / 2, r2_scores, width, label='R² Score', alpha=0.7)

axes[0, 1].set_xlabel('模型')
axes[0, 1].set_ylabel('分数')
axes[0, 1].set_title('模型性能比较')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(model_names, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom')

# 5.3 特征重要性（系数比较）
feature_names = diabetes.feature_names
coefficients_data = []

for name in ['Linear Regression', 'Ridge (alpha=1.0)', 'Lasso (alpha=0.1)']:
    if results[name]['coefficients'] is not None:
        for feature, coef in zip(feature_names, results[name]['coefficients']):
            coefficients_data.append({'Model': name, 'Feature': feature, 'Coefficient': coef})

import pandas as pd

coef_df = pd.DataFrame(coefficients_data)

pivot_df = coef_df.pivot(index='Feature', columns='Model', values='Coefficient')
pivot_df.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('不同模型的特征系数比较')
axes[1, 0].set_ylabel('系数值')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 5.4 残差分析
residuals = y_test - best_result['predictions']
axes[1, 1].scatter(best_result['predictions'], residuals, alpha=0.7, color='green')
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('预测值')
axes[1, 1].set_ylabel('残差')
axes[1, 1].set_title(f'残差分析 ({best_model_name})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 详细结果分析
print("\n" + "=" * 50)

# 找到最佳模型
best_model = min(results.items(), key=lambda x: x[1]['mse'])
print(f"最佳模型: {best_model[0]}")
print(f"最佳MSE: {best_model[1]['mse']:.2f}")
print(f"最佳R2: {best_model[1]['r2']:.3f}")

# 系数分析
print("\n特征系数分析:")
linear_coef = results['Linear Regression']['coefficients']
for feature, coef in zip(feature_names, linear_coef):
    print(f"  {feature:15}: {coef:7.3f}")

# 正则化效果分析
print("\n正则化效果分析:")

# 显示拉索回归的特征选择效果
lasso_coef = results['Lasso (alpha=1.0)']['coefficients']
selected_features = [feature for feature, coef in zip(feature_names, lasso_coef) if abs(coef) > 0.01]
print(f"Lasso (alpha=1.0) 选择的特征数: {len(selected_features)}/{len(feature_names)}")
print(f"选择的特征: {selected_features}")

# 7. 数据标准化效果验证
print("\n" + "=" * 30)

# 比较标准化前后的效果
lr_no_scale = LinearRegression()
lr_no_scale.fit(X_train, y_train)
y_pred_no_scale = lr_no_scale.predict(X_test)
mse_no_scale = mean_squared_error(y_test, y_pred_no_scale)
r2_no_scale = r2_score(y_test, y_pred_no_scale)

print(f"未标准化 - MSE: {mse_no_scale:.2f}, R²: {r2_no_scale:.3f}")
print(f"标准化后 - MSE: {results['Linear Regression']['mse']:.2f}, R²: {results['Linear Regression']['r2']:.3f}")

if results['Linear Regression']['mse'] < mse_no_scale:
    print("标准化提高了模型性能")
else:
    print("标准化效果不明显")
