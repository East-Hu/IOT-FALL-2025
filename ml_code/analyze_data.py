#!/usr/bin/env python3
"""
详细数据分析脚本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix

print("=" * 70)
print("详细数据分析")
print("=" * 70)

# 1. 加载数据
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1)
y = df['label'].apply(lambda x: int(float(x)) if pd.notna(x) else x)

print(f"\n总样本数: {len(df)}")
print(f"特征数量: {len(X.columns)}")

# 2. 数据分布
print("\n" + "=" * 70)
print("每个数字的样本数")
print("=" * 70)
for digit in sorted(y.unique()):
    count = sum(y == digit)
    percentage = count / len(y) * 100
    print(f"数字 {digit}: {count:4d} 个样本 ({percentage:5.2f}%)")

# 3. 加载最新模型并分析混淆情况
print("\n" + "=" * 70)
print("混淆矩阵分析（加载最新XGBoost模型）")
print("=" * 70)

import glob
model_files = sorted(glob.glob('models/xgboost_*.pkl'))
if model_files:
    latest_model = model_files[-1]
    print(f"使用模型: {latest_model}")

    # 加载模型
    model_data = joblib.load(latest_model)
    model = model_data['model']
    scaler = model_data['scaler']

    # 处理特征
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    # 划分测试集（使用相同的random_state）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    X_test_scaled = scaler.transform(X_test)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    print("\n混淆矩阵（行=实际，列=预测）:")
    print("    ", end="")
    for i in range(10):
        print(f"{i:4d}", end=" ")
    print()
    print("    " + "-" * 50)

    for i in range(10):
        print(f"{i:2d} |", end=" ")
        for j in range(10):
            if i == j:
                # 对角线（正确预测）用粗体
                print(f"\033[1m{cm[i,j]:4d}\033[0m", end=" ")
            elif cm[i,j] > 0:
                # 错误预测
                print(f"{cm[i,j]:4d}", end=" ")
            else:
                print("   .", end=" ")
        print()

    # 找出最容易混淆的数字对
    print("\n" + "=" * 70)
    print("最容易混淆的数字对（前10个）")
    print("=" * 70)

    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i,j] > 0:
                confusions.append((i, j, cm[i,j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    for i, (actual, predicted, count) in enumerate(confusions[:10], 1):
        total_actual = sum(cm[actual, :])
        error_rate = count / total_actual * 100
        print(f"{i:2d}. 实际={actual}, 预测成{predicted}: {count}次 ({error_rate:.1f}%)")

    # 分析每个数字的准确率
    print("\n" + "=" * 70)
    print("每个数字的准确率")
    print("=" * 70)

    for digit in range(10):
        total = sum(cm[digit, :])
        correct = cm[digit, digit]
        accuracy = correct / total * 100 if total > 0 else 0

        # 找出这个数字最容易被误判为哪个数字
        errors = [(j, cm[digit, j]) for j in range(10) if j != digit and cm[digit, j] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)

        print(f"数字 {digit}: {correct}/{total} = {accuracy:.1f}%", end="")
        if errors:
            top_error = errors[0]
            print(f"  (最常误判为 {top_error[0]}, {top_error[1]}次)")
        else:
            print()

    # 整体准确率
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    overall_accuracy = total_correct / total_samples * 100

    print("\n" + "=" * 70)
    print(f"整体准确率: {total_correct}/{total_samples} = {overall_accuracy:.2f}%")
    print("=" * 70)

print("\n" + "=" * 70)
print("数据质量建议")
print("=" * 70)

# 检查样本数是否足够
if len(df) < 1000:
    print("⚠ 样本数较少（< 1000），建议继续收集到2000+")
elif len(df) < 2000:
    print("✓ 样本数适中（1000-2000），可以考虑继续收集")
else:
    print("✓ 样本数充足（> 2000）")

# 检查数据平衡性
counts = y.value_counts()
imbalance = counts.max() / counts.min()
if imbalance < 1.2:
    print("✓ 数据平衡良好")
elif imbalance < 1.5:
    print("⚠ 数据略有不平衡，可以补充样本少的数字")
else:
    print("✗ 数据不平衡！需要补充样本少的数字")

print("\n建议：")
if overall_accuracy < 60:
    print("1. 准确率较低（< 60%），主要问题可能是：")
    print("   - 数据质量差（按键不稳定）")
    print("   - 持握方式变化大")
    print("   - 按键速度差异大")
elif overall_accuracy < 75:
    print("1. 准确率中等（60-75%），这是正常水平")
    print("   - 可以通过增加数据到2500+样本来提升到75-80%")
    print("   - 关注最容易混淆的数字对")
else:
    print("1. 准确率良好（> 75%），已经接近理论上限")

print("2. 针对最容易混淆的数字对：")
print("   - 专门收集这些数字的数据")
print("   - 注意按键时的姿势一致性")
print("3. 数据收集建议：")
print("   - 保持持握方式一致")
print("   - 按键速度保持稳定")
print("   - 避免过快或过慢")
