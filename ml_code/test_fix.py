#!/usr/bin/env python3
"""
快速测试：验证重复数字识别修复是否有效
"""

import sys
import pandas as pd
import numpy as np
from generate_synthetic_data_v2 import analyze_normal_data, generate_keypress

# 1. 生成一个测试文件（11111）
print("=" * 70)
print("快速测试：重复数字识别修复")
print("=" * 70)

print("\n步骤 1: 加载统计数据...")
stats = analyze_normal_data()

print("\n步骤 2: 生成测试密码 11111...")
current_timestamp = 1762000000000
current_nanotime = 8100000000000

keypresses = []

for i in range(5):
    digit = 1
    print(f"  生成第 {i+1} 个数字 1...")

    keypress_df, duration = generate_keypress(digit, stats, current_timestamp, current_nanotime)
    keypresses.append(keypress_df)

    # 增大间隔，确保能分割
    interval = 700  # >500ms阈值
    current_timestamp += duration + interval
    current_nanotime += (duration + interval) * 1000000

    # 添加间隔数据
    if i < 4:  # 最后一个不需要间隔
        inter_rows = []
        for sensor_type in ['ACC', 'MAG', 'GYRO', 'ROT_VEC']:
            inter_rows.append({
                'timestamp': int(current_timestamp - interval/2),
                'nanoTime': int(current_nanotime - interval*1000000/2),
                'type': sensor_type,
                'x': np.random.randn() * 0.1,
                'y': np.random.randn() * 0.1,
                'z': np.random.randn() * 0.1,
                'w': 0.0,
                'label': ''
            })
        keypresses.append(pd.DataFrame(inter_rows))

# 合并
combined = pd.concat(keypresses, ignore_index=True)
test_file = '../test_data/test_quick_11111.csv'
combined.to_csv(test_file, index=False)
print(f"  ✓ 测试文件已保存: {test_file}")

# 2. 测试预处理
print("\n步骤 3: 测试预处理（分割检测）...")
import importlib.util
import os
spec = importlib.util.spec_from_file_location("data_preprocessing",
                                                os.path.join(os.path.dirname(__file__), "1_data_preprocessing.py"))
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)
SensorDataProcessor = data_preprocessing.SensorDataProcessor

processor = SensorDataProcessor(sampling_rate=50)
df = pd.read_csv(test_file)
segments = processor.segment_by_label(df)

print(f"\n检测到 {len(segments)} 个按键事件")
print("-" * 70)

for i, seg in enumerate(segments, 1):
    label = seg['label']
    duration = seg['data']['timestamp'].max() - seg['data']['timestamp'].min()
    print(f"  按键 {i}: 数字={label}, 持续时间={duration:.0f}ms")

print("-" * 70)

# 3. 评估结果
if len(segments) == 5:
    print("\n✓ 测试通过！成功识别5个重复的数字1")
    print("修复生效：预处理能够正确分割相同label的按键")
elif len(segments) == 1:
    print("\n✗ 测试失败！只识别到1个按键")
    print("问题依然存在：预处理无法分割相同label的按键")
    print("\n可能原因：")
    print("  1. 时间间隔不够大（需要>500ms）")
    print("  2. 预处理代码未正确更新")
else:
    print(f"\n⚠ 部分成功：识别到{len(segments)}个按键（预期5个）")

print("\n" + "=" * 70)
