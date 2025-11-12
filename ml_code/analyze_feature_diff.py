#!/usr/bin/env python3
"""
分析真实数据和合成数据的特征差异
"""

import pandas as pd
import numpy as np
import sys
import os

# 导入特征提取器
import importlib.util
spec = importlib.util.spec_from_file_location("feature_extraction",
                                                os.path.join(os.path.dirname(__file__), "2_feature_extraction.py"))
feature_extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_extraction)
FeatureExtractor = feature_extraction.FeatureExtractor

# 导入数据预处理
spec2 = importlib.util.spec_from_file_location("data_preprocessing",
                                                os.path.join(os.path.dirname(__file__), "1_data_preprocessing.py"))
data_preprocessing = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(data_preprocessing)
SensorDataProcessor = data_preprocessing.SensorDataProcessor

print("=" * 70)
print("特征分布差异分析")
print("=" * 70)

# 1. 处理真实数据
print("\n提取真实数据特征...")
real_file = '../sensor_data/files/password_training_1761878641118.csv'
df_real = pd.read_csv(real_file)

processor = SensorDataProcessor()
segments_real = processor.segment_by_label(df_real)

print(f"真实数据识别到 {len(segments_real)} 个按键")

extractor = FeatureExtractor()
real_features = []
for seg in segments_real:
    features = extractor.extract_keypress_features(seg['data'])
    real_features.append(features)

df_real_features = pd.DataFrame(real_features)

# 2. 处理合成数据（从测试数据中取）
print("\n提取合成数据特征...")
synth_file = '../test_data/test_password_24680_1762002104823.csv'
df_synth = pd.read_csv(synth_file)

segments_synth = processor.segment_by_label(df_synth)
print(f"合成数据识别到 {len(segments_synth)} 个按键")

synth_features = []
for seg in segments_synth:
    features = extractor.extract_keypress_features(seg['data'])
    synth_features.append(features)

df_synth_features = pd.DataFrame(synth_features)

# 3. 比较特征分布
print("\n" + "=" * 70)
print("关键特征对比（前10个最重要特征）")
print("=" * 70)

important_features = [
    'ROT_VEC_w_mean', 'GYRO_z_mean', 'GYRO_y_rms',
    'MAG_z_rms', 'ACC_y_std', 'MAG_y_mean',
    'GYRO_x_rms', 'GYRO_y_mean', 'ACC_y_q25', 'ACC_x_rms'
]

print(f"\n{'特征名':<30} {'真实数据均值':<15} {'合成数据均值':<15} {'差异倍数':<10}")
print("-" * 70)

large_diff_features = []

for feat in important_features:
    if feat in df_real_features.columns and feat in df_synth_features.columns:
        real_mean = df_real_features[feat].mean()
        synth_mean = df_synth_features[feat].mean()

        # 计算差异
        if synth_mean != 0:
            ratio = abs(real_mean / synth_mean)
        else:
            ratio = 999

        if ratio > 2 or ratio < 0.5:
            large_diff_features.append((feat, ratio))

        print(f"{feat:<30} {real_mean:>14.4f} {synth_mean:>14.4f} {ratio:>9.2f}x")

print("\n" + "=" * 70)
print("差异诊断")
print("=" * 70)

if len(large_diff_features) > 0:
    print(f"\n发现 {len(large_diff_features)} 个特征差异较大（>2倍或<0.5倍）")
    print("\n这就是为什么模型无法识别真实数据！")
    print("\n解决方案：需要用真实数据微调模型，或生成更真实的合成数据")
else:
    print("\n特征分布相似，问题可能在其他地方")

print("\n" + "=" * 70)
print("数据质量检查")
print("=" * 70)

# 检查真实数据的统计
print("\n真实数据统计:")
for i, seg in enumerate(segments_real):
    label = seg['label']
    duration = seg['data']['timestamp'].max() - seg['data']['timestamp'].min()
    rows = len(seg['data'])
    print(f"  按键 {i+1}: 数字={int(float(label))}, 持续={duration:.0f}ms, 数据行={rows}")

print("\n合成数据统计:")
for i, seg in enumerate(segments_synth):
    label = seg['label']
    duration = seg['data']['timestamp'].max() - seg['data']['timestamp'].min()
    rows = len(seg['data'])
    print(f"  按键 {i+1}: 数字={int(float(label))}, 持续={duration:.0f}ms, 数据行={rows}")
