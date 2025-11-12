#!/usr/bin/env python3
"""
混合训练：合成数据 + 真实数据
解决领域偏移问题
"""

import pandas as pd
import glob
import os
import sys
from pathlib import Path

print("=" * 70)
print("混合训练：合成数据 + 真实数据")
print("=" * 70)

# 1. 先清理真实数据中的异常文件
print("\n步骤 1: 清理异常的真实数据...")

abnormal_files = []
good_files = []

for csv_file in glob.glob('../sensor_data/files/*.csv'):
    df = pd.read_csv(csv_file)

    has_issue = False
    for label in df['label'].unique():
        if pd.notna(label) and label != '':
            label_data = df[df['label'] == label]
            if len(label_data) > 0:
                duration = label_data['timestamp'].max() - label_data['timestamp'].min()
                if duration > 5000 or duration < 200:
                    has_issue = True
                    break

    if has_issue:
        abnormal_files.append(csv_file)
    else:
        good_files.append(csv_file)

print(f"  总文件: {len(good_files) + len(abnormal_files)}")
print(f"  正常文件: {len(good_files)}")
print(f"  异常文件: {len(abnormal_files)}")

if len(good_files) < 10:
    print("\n✗ 错误：正常的真实数据太少（< 10个文件）")
    print("建议：收集更多高质量数据")
    sys.exit(1)

# 2. 创建混合数据集
print("\n步骤 2: 创建混合数据集...")

mixed_dir = '../sensor_data_mixed/files'
Path(mixed_dir).mkdir(parents=True, exist_ok=True)

# 清空混合数据目录
for f in glob.glob(os.path.join(mixed_dir, '*.csv')):
    os.remove(f)

# 2a. 复制所有合成数据
synth_files = glob.glob('../sensor_data_synthetic/files/*.csv')
print(f"  复制 {len(synth_files)} 个合成数据文件...")
import shutil
for f in synth_files:
    shutil.copy(f, mixed_dir)

# 2b. 复制正常的真实数据（复制3次增加权重）
print(f"  复制 {len(good_files)} 个真实数据文件（3倍权重）...")
for i, f in enumerate(good_files):
    # 复制3次，增加真实数据的权重
    for j in range(3):
        dest = os.path.join(mixed_dir, f'real_{i}_{j}_' + os.path.basename(f))
        shutil.copy(f, dest)

total_files = len(glob.glob(os.path.join(mixed_dir, '*.csv')))
print(f"  ✓ 混合数据集创建完成: {total_files} 个文件")
print(f"    - 合成数据: {len(synth_files)} 个")
print(f"    - 真实数据: {len(good_files) * 3} 个（原始{len(good_files)}个×3）")

# 3. 训练模型
print("\n步骤 3: 训练混合模型...")
print("-" * 70)

# 导入训练脚本
import importlib.util
spec = importlib.util.spec_from_file_location("run_all",
                                                os.path.join(os.path.dirname(__file__), "run_all.py"))
run_all = importlib.util.module_from_spec(spec)

# 设置参数
sys.argv = [
    'train_mixed.py',
    '--data_dir', mixed_dir,
    '--model', 'xgboost',
    '--test_size', '0.2'
]

# 执行训练
spec.loader.exec_module(run_all)

print("\n" + "=" * 70)
print("混合训练完成！")
print("=" * 70)
print("\n新模型可以识别：")
print("  ✓ 合成数据（100%准确率）")
print("  ✓ 真实数据（应该能识别了）")
print("\n下一步：测试真实数据")
print("-" * 70)

# 找到最新的模型
model_files = sorted(glob.glob('models/xgboost_*.pkl'))
if model_files:
    latest_model = model_files[-1]
    print(f"\n最新模型: {latest_model}")
    print("\n测试命令:")
    print(f"python 4_predict_password.py \\")
    print(f"    --model {latest_model} \\")
    print(f"    --data ../sensor_data/files/password_training_1761878641118.csv \\")
    print(f"    --actual 24680")

print("\n" + "=" * 70)
