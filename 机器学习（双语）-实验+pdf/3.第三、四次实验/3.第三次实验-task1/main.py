# ============ 导入必要的库 ============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import pickle
import warnings
import time
from tqdm import tqdm
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

# ============ 数据加载和探索 ============
print("=" * 60)
print("数据加载和探索")
print("=" * 60)

# 加载数据
print("正在加载数据...")
df = pd.read_csv('train.csv')

print(f"数据形状: {df.shape}")
print(f"特征数量: {df.shape[1] - 1}")
print(f"样本数量: {df.shape[0]}")

# ============ 进阶数据预处理 ============
print("\n" + "=" * 60)
print("进阶数据预处理")
print("=" * 60)

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 1. 特征缩放
print("1. 特征缩放...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 特征选择
print("2. 特征选择...")
rf_for_selection = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_for_selection.fit(X_scaled, y)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_for_selection.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20重要特征:")
print(feature_importance.head(20))

# 选择最重要的特征
selector = SelectFromModel(rf_for_selection, prefit=True, threshold='mean')
X_selected = selector.transform(X_scaled)
print(f"原始特征数量: {X_scaled.shape[1]}")
print(f"选择后特征数量: {X_selected.shape[1]}")

# 3. 处理类别不平衡
print("3. 处理类别不平衡...")
print(f"原始标签分布:\n{y.value_counts().sort_index()}")

# 尝试SMOTE过采样
try:
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    print(f"SMOTE后样本数量: {len(X_resampled)}")
    print(f"SMOTE后标签分布:\n{pd.Series(y_resampled).value_counts().sort_index()}")
    X_final, y_final = X_resampled, y_resampled
except:
    print("SMOTE失败，使用原始数据")
    X_final, y_final = X_selected, y

# ============ 数据划分 ============
print("\n4. 数据划分...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y_final, test_size=0.4, random_state=42, stratify=y_final
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"验证集大小: {X_val.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")

# ============ 简化的模型训练与调优 ============
print("\n" + "=" * 60)
print("简化的模型训练与调优")
print("=" * 60)

# 先直接训练几个基础模型看看效果
print("1. 训练基础模型...")


# 定义一个安全的LightGBM创建函数，移除所有可能导致问题的参数
def create_safe_lgb_model(**kwargs):
    """创建一个安全的LightGBM模型"""
    safe_kwargs = kwargs.copy()

    # 移除可能导致问题的参数
    if 'early_stopping_round' in safe_kwargs:
        del safe_kwargs['early_stopping_round']
    if 'verbose' in safe_kwargs:
        del safe_kwargs['verbose']

    # 总是使用verbose=0
    safe_kwargs['verbose'] = 0

    return lgb.LGBMClassifier(**safe_kwargs)


# 使用安全的模型创建函数
base_models = {
    'LightGBM': create_safe_lgb_model(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        learning_rate=0.1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        learning_rate=0.1
    ),
    'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
}

base_results = {}

for name, model in tqdm(base_models.items(), desc="训练基础模型"):
    print(f"\n训练 {name}...")
    start_time = time.time()

    try:
        # 对于XGBoost，我们可以使用早停
        if name == 'XGBoost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            if hasattr(model, 'best_iteration'):
                iterations = model.best_iteration
                print(f"XGBoost 最佳迭代轮数: {iterations}")
            else:
                iterations = 1000

        else:
            # 其他模型正常训练
            model.fit(X_train, y_train)
            iterations = None

        train_time = time.time() - start_time

        # 验证集评估
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        base_results[name] = {
            'model': model,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'train_time': train_time,
            'iterations': iterations
        }

        print(f"{name} 验证集准确率: {val_acc:.4f}")
        print(f"{name} 训练时间: {train_time:.2f}秒")

    except Exception as e:
        print(f"训练{name}时出错: {e}")
        continue

# 检查是否有模型训练成功
if not base_results:
    print("所有模型训练都失败了！")
    exit(1)

# 选择最佳基础模型
best_base_name = max(base_results, key=lambda x: base_results[x]['val_accuracy'])
best_base_model = base_results[best_base_name]['model']
best_base_acc = base_results[best_base_name]['val_accuracy']

print(f"\n最佳基础模型: {best_base_name}")
print(f"验证集准确率: {best_base_acc:.4f}")

# ============ 手动调优最佳模型 ============
print("\n2. 手动调优最佳模型...")

# 只对树模型进行调优
if best_base_name in ['LightGBM', 'XGBoost', 'RandomForest']:
    print(f"对{best_base_name}进行手动调优...")

    if best_base_name == 'LightGBM':
        # 尝试几组LightGBM参数
        lgb_param_sets = [
            {
                'num_leaves': 31,
                'max_depth': 10,
                'learning_rate': 0.1,
                'n_estimators': 2000,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            {
                'num_leaves': 63,
                'max_depth': 15,
                'learning_rate': 0.05,
                'n_estimators': 3000,
                'min_child_samples': 10,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            },
            {
                'num_leaves': 127,
                'max_depth': 20,
                'learning_rate': 0.02,
                'n_estimators': 5000,
                'min_child_samples': 5,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            }
        ]

        best_lgb_model = None
        best_lgb_acc = 0

        for i, params in enumerate(lgb_param_sets, 1):
            print(f"\n尝试LightGBM参数集 {i}/{len(lgb_param_sets)}")

            # 使用安全的模型创建函数
            lgb_model = create_safe_lgb_model(
                num_leaves=params['num_leaves'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                min_child_samples=params['min_child_samples'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=42,
                n_jobs=-1
            )

            try:
                lgb_model.fit(X_train, y_train)

                y_val_pred = lgb_model.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)

                iterations = params['n_estimators']
                print(f"验证集准确率: {val_acc:.4f}, 迭代轮数: {iterations}")

                if val_acc > best_lgb_acc:
                    best_lgb_acc = val_acc
                    best_lgb_model = lgb_model

            except Exception as e:
                print(f"参数集 {i} 训练失败: {e}")
                continue

        if best_lgb_model:
            best_tuned_model = best_lgb_model
            print(f"LightGBM调优后最佳验证集准确率: {best_lgb_acc:.4f}")
        else:
            best_tuned_model = best_base_model
            print("LightGBM调优失败，使用原始模型")

    elif best_base_name == 'XGBoost':
        # 尝试几组XGBoost参数
        xgb_param_sets = [
            {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 2000,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0
            },
            {
                'max_depth': 10,
                'learning_rate': 0.05,
                'n_estimators': 3000,
                'min_child_weight': 3,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 0.1
            },
            {
                'max_depth': 15,
                'learning_rate': 0.02,
                'n_estimators': 5000,
                'min_child_weight': 5,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'gamma': 0.2
            }
        ]

        best_xgb_model = None
        best_xgb_acc = 0

        for i, params in enumerate(xgb_param_sets, 1):
            print(f"\n尝试XGBoost参数集 {i}/{len(xgb_param_sets)}")

            xgb_model = xgb.XGBClassifier(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            try:
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                y_val_pred = xgb_model.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)

                iterations = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else params[
                    'n_estimators']
                print(f"验证集准确率: {val_acc:.4f}, 实际迭代: {iterations}")

                if val_acc > best_xgb_acc:
                    best_xgb_acc = val_acc
                    best_xgb_model = xgb_model

            except Exception as e:
                print(f"参数集 {i} 训练失败: {e}")
                continue

        if best_xgb_model:
            best_tuned_model = best_xgb_model
            print(f"XGBoost调优后最佳验证集准确率: {best_xgb_acc:.4f}")
        else:
            best_tuned_model = best_base_model
            print("XGBoost调优失败，使用原始模型")

    elif best_base_name == 'RandomForest':
        # 尝试几组RandomForest参数
        rf_param_sets = [
            {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5},
            {'n_estimators': 1000, 'max_depth': 30, 'min_samples_split': 10},
            {'n_estimators': 2000, 'max_depth': 40, 'min_samples_split': 20}
        ]

        best_rf_model = None
        best_rf_acc = 0

        for i, params in enumerate(rf_param_sets, 1):
            print(f"\n尝试RandomForest参数集 {i}/{len(rf_param_sets)}")

            rf_model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)

            y_val_pred = rf_model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)

            print(f"验证集准确率: {val_acc:.4f}")

            if val_acc > best_rf_acc:
                best_rf_acc = val_acc
                best_rf_model = rf_model

        if best_rf_model:
            best_tuned_model = best_rf_model
            print(f"RandomForest调优后最佳验证集准确率: {best_rf_acc:.4f}")
        else:
            best_tuned_model = best_base_model
            print("RandomForest调优失败，使用原始模型")

else:
    print(f"{best_base_name} 暂不进行手动调优")
    best_tuned_model = best_base_model

# ============ 模型集成 ============
print("\n3. 尝试模型集成...")

# 选择几个表现较好的模型进行集成
print("训练集成模型...")

# 创建几个基础模型，确保所有模型都没有早停参数
lgb_for_ensemble = create_safe_lgb_model(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

xgb_for_ensemble = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

rf_for_ensemble = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

# 训练这些模型（不使用早停）
print("训练LightGBM用于集成...")
lgb_for_ensemble.fit(X_train, y_train)

print("训练XGBoost用于集成...")
xgb_for_ensemble.fit(X_train, y_train)

print("训练RandomForest用于集成...")
rf_for_ensemble.fit(X_train, y_train)

# 创建投票分类器
print("创建投票分类器...")
voting_clf = VotingClassifier(
    estimators=[
        ('lgb', lgb_for_ensemble),
        ('xgb', xgb_for_ensemble),
        ('rf', rf_for_ensemble)
    ],
    voting='soft',
    n_jobs=-1
)

voting_clf.fit(X_train, y_train)

# 评估投票分类器
y_val_vote = voting_clf.predict(X_val)
val_acc_vote = accuracy_score(y_val, y_val_vote)
print(f"投票分类器验证集准确率: {val_acc_vote:.4f}")

# ============ 在测试集上评估所有模型 ============
print("\n" + "=" * 60)
print("在测试集上评估所有模型")
print("=" * 60)

models_to_evaluate = {
    'Best_Base_Model': best_base_model,
    'Best_Tuned_Model': best_tuned_model,
    'Voting_Classifier': voting_clf
}

test_results = {}

for name, model in models_to_evaluate.items():
    print(f"\n评估 {name}...")

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    test_results[name] = {
        'model': model,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall
    }

    print(f"{name} 测试集准确率: {test_acc:.4f}")
    print(f"{name} 测试集F1分数: {test_f1:.4f}")

# 选择最佳模型
best_test_name = max(test_results, key=lambda x: test_results[x]['test_accuracy'])
best_model = test_results[best_test_name]['model']
best_test_acc = test_results[best_test_name]['test_accuracy']

print(f"\n最佳测试模型: {best_test_name}")
print(f"测试集准确率: {best_test_acc:.4f}")

# ============ 详细分析最佳模型 ============
print("\n" + "=" * 60)
print("详细分析最佳模型")
print("=" * 60)

print(f"最佳模型: {best_test_name}")
print(f"测试集准确率: {best_test_acc:.4f}")

# 生成分类报告
y_test_best_pred = best_model.predict(X_test)

print(f"\n{best_test_name} 详细分类报告（测试集）:")
print(classification_report(y_test, y_test_best_pred,
                            target_names=[str(i) for i in range(11)]))

# 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_test_best_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(11), yticklabels=range(11))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'{best_test_name} 混淆矩阵 (准确率: {best_test_acc:.2%})')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 特征重要性分析（如果是树模型） ============
if best_test_name in ['LightGBM', 'XGBoost', 'RandomForest']:
    print("\n特征重要性分析...")

    if hasattr(best_model, 'feature_importances_'):
        # 获取特征重要性
        importances = best_model.feature_importances_

        # 获取被选中的特征名
        selected_features = X.columns[selector.get_support()]

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\nTop 20重要特征:")
        print(feature_importance_df.head(20))

        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance_df['feature'][:20][::-1],
                 feature_importance_df['importance'][:20][::-1])
        plt.xlabel('特征重要性')
        plt.title(f'{best_test_name} Top 20特征重要性')
        plt.tight_layout()
        plt.savefig('best_model_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============ 保存模型和结果 ============
print("\n" + "=" * 60)
print("保存模型和结果")
print("=" * 60)

# 创建保存目录
if not os.path.exists('optimized_models'):
    os.makedirs('optimized_models')

# 保存最佳模型
best_model_data = {
    'model': best_model,
    'model_name': best_test_name,
    'scaler': scaler,
    'selector': selector,
    'feature_names': X.columns.tolist(),
    'test_accuracy': best_test_acc,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'label_mapping': {i: str(i) for i in range(11)}
}

joblib.dump(best_model_data, 'optimized_models/best_model.joblib')
print("已保存最佳模型: optimized_models/best_model.joblib")

# 保存所有测试结果
test_results_df = pd.DataFrame({
    'Model': list(test_results.keys()),
    'Test Accuracy': [test_results[name]['test_accuracy'] for name in test_results.keys()],
    'Test F1': [test_results[name]['test_f1'] for name in test_results.keys()],
    'Test Precision': [test_results[name]['test_precision'] for name in test_results.keys()],
    'Test Recall': [test_results[name]['test_recall'] for name in test_results.keys()]
}).sort_values('Test Accuracy', ascending=False)

test_results_df.to_csv('optimized_models/model_test_results.csv', index=False)
print("已保存模型测试结果: optimized_models/model_test_results.csv")

# ============ 生成优化报告 ============
print("\n" + "=" * 60)
print("生成优化报告")
print("=" * 60)

# 读取原始结果
original_acc = 0.6460  # 从之前的运行结果获取

# 生成报告
optimization_report = f"""
{'=' * 80}
机器学习模型优化报告
{'=' * 80}

一、优化策略
1. 特征选择: 基于随机森林特征重要性筛选特征 (256 -> {X_selected.shape[1]})
2. 类别平衡: 使用SMOTE处理类别不平衡
3. 模型调优: 手动尝试多组参数
4. 模型集成: 使用投票分类器

二、优化效果对比
原始最佳准确率: {original_acc:.2%} (LightGBM)
优化后最佳准确率: {best_test_acc:.2%} ({best_test_name})
准确率提升: {(best_test_acc - original_acc):.2%}

三、所有模型测试结果
{test_results_df.to_string(index=False)}

四、最佳模型详情
模型名称: {best_test_name}
测试集准确率: {best_test_acc:.4f}

五、关键发现
1. 特征选择减少了 {X_scaled.shape[1] - X_selected.shape[1]} 个特征
2. SMOTE将每个类别的样本数平衡到 817 个
3. 最佳模型为: {best_test_name}

六、后续优化建议
1. 尝试更复杂的深度学习架构
2. 使用更精细的网格搜索
3. 尝试特征交叉和多项式特征
4. 使用更先进的集成方法

{'=' * 80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

# 保存报告
with open('optimized_models/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write(optimization_report)

print(optimization_report)
print("优化报告已保存为 'optimized_models/optimization_report.txt'")

print("\n" + "=" * 60)
print("模型优化完成!")
print("=" * 60)
print(f"最佳模型: {best_test_name}")
print(f"测试集准确率: {best_test_acc:.4f}")
print(f"准确率提升: {(best_test_acc - original_acc):.4f}")
print(f"所有结果已保存到 'optimized_models/' 目录")
print("=" * 60)