#!/usr/bin/env python3
"""
生成合成的高质量传感器数据（匹配原始数据格式）
格式：timestamp, nanoTime, type, x, y, z, w, label
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from collections import defaultdict

def analyze_normal_data():
    """分析正常数据的统计特征"""
    print("=" * 70)
    print("分析正常数据...")
    print("=" * 70)

    # 按 (digit, sensor_type) 收集数据
    stats = defaultdict(lambda: {'x': [], 'y': [], 'z': [], 'w': [], 'durations': []})

    for csv_file in glob.glob('../sensor_data/files/*.csv'):
        df = pd.read_csv(csv_file)

        for label in df['label'].unique():
            if pd.notna(label) and label != '':
                label_data = df[df['label'] == label].copy()

                if len(label_data) > 0:
                    duration = label_data['timestamp'].max() - label_data['timestamp'].min()

                    # 只使用正常数据（1-5秒）
                    if 1000 <= duration <= 5000:
                        digit = int(float(label))
                        stats[(digit, 'all')]['durations'].append(duration)

                        # 分传感器类型收集数据
                        for sensor_type in ['ACC', 'MAG', 'GYRO', 'ROT_VEC']:
                            sensor_data = label_data[label_data['type'] == sensor_type]
                            if len(sensor_data) > 0:
                                stats[(digit, sensor_type)]['x'].extend(sensor_data['x'].dropna())
                                stats[(digit, sensor_type)]['y'].extend(sensor_data['y'].dropna())
                                stats[(digit, sensor_type)]['z'].extend(sensor_data['z'].dropna())
                                if sensor_type == 'ROT_VEC':
                                    stats[(digit, sensor_type)]['w'].extend(sensor_data['w'].dropna())

    # 计算统计量
    computed_stats = {}
    for (digit, sensor), data in stats.items():
        if sensor == 'all':
            computed_stats[(digit, sensor)] = {
                'duration_mean': np.mean(data['durations']),
                'duration_std': np.std(data['durations'])
            }
        else:
            computed_stats[(digit, sensor)] = {
                'x_mean': np.mean(data['x']) if data['x'] else 0,
                'x_std': np.std(data['x']) if data['x'] else 0.1,
                'y_mean': np.mean(data['y']) if data['y'] else 0,
                'y_std': np.std(data['y']) if data['y'] else 0.1,
                'z_mean': np.mean(data['z']) if data['z'] else 0,
                'z_std': np.std(data['z']) if data['z'] else 0.1,
            }
            if sensor == 'ROT_VEC' and data['w']:
                computed_stats[(digit, sensor)]['w_mean'] = np.mean(data['w'])
                computed_stats[(digit, sensor)]['w_std'] = np.std(data['w'])

    print(f"\n找到 {len([k for k in computed_stats if k[1] == 'all'])} 个数字的统计数据")
    return computed_stats

def generate_sensor_sequence(digit, sensor_type, stats, duration_ms, sampling_rate=50):
    """生成单个传感器类型的时间序列"""

    key = (digit, sensor_type)
    if key not in stats:
        # 使用默认值
        x_mean, y_mean, z_mean = 0, 0, 0
        x_std, y_std, z_std = 1, 1, 1
        w_mean, w_std = 0, 0.1
    else:
        s = stats[key]
        x_mean, x_std = s.get('x_mean', 0), s.get('x_std', 0.1)
        y_mean, y_std = s.get('y_mean', 0), s.get('y_std', 0.1)
        z_mean, z_std = s.get('z_mean', 0), s.get('z_std', 0.1)
        w_mean, w_std = s.get('w_mean', 0), s.get('w_std', 0.1)

    # 计算样本数
    num_samples = int(duration_ms * sampling_rate / 1000)

    # 生成时间序列
    x_values = np.random.normal(x_mean, x_std * 0.8, num_samples)
    y_values = np.random.normal(y_mean, y_std * 0.8, num_samples)
    z_values = np.random.normal(z_mean, z_std * 0.8, num_samples)

    # 添加平滑趋势
    t = np.linspace(0, duration_ms/1000, num_samples)
    freq = np.random.uniform(0.5, 2.0)
    x_values += x_std * 0.2 * np.sin(2 * np.pi * freq * t)
    y_values += y_std * 0.2 * np.sin(2 * np.pi * freq * t + np.pi/3)
    z_values += z_std * 0.2 * np.sin(2 * np.pi * freq * t + 2*np.pi/3)

    if sensor_type == 'ROT_VEC':
        w_values = np.random.normal(w_mean, w_std * 0.8, num_samples)
        w_values += w_std * 0.2 * np.sin(2 * np.pi * freq * t + np.pi/2)
    else:
        w_values = np.zeros(num_samples)

    return x_values, y_values, z_values, w_values

def generate_keypress(digit, stats, start_timestamp, start_nanotime, sampling_rate=50):
    """生成单个按键的完整传感器数据"""

    # 生成持续时间（1-3秒）
    duration_key = (digit, 'all')
    if duration_key in stats:
        duration_mean = stats[duration_key]['duration_mean']
        duration_std = stats[duration_key]['duration_std']
        duration = np.random.normal(duration_mean, duration_std * 0.5)
    else:
        duration = np.random.uniform(1500, 2500)

    duration = np.clip(duration, 1000, 3000)  # 限制在1-3秒

    rows = []

    # 为每个传感器类型生成数据
    for sensor_type in ['ACC', 'MAG', 'GYRO', 'ROT_VEC']:
        x_vals, y_vals, z_vals, w_vals = generate_sensor_sequence(
            digit, sensor_type, stats, duration, sampling_rate
        )

        num_samples = len(x_vals)

        # 生成时间戳（均匀分布）
        timestamps = np.linspace(start_timestamp, start_timestamp + duration, num_samples)
        nanotimes = np.linspace(start_nanotime, start_nanotime + duration * 1000000, num_samples)

        for i in range(num_samples):
            rows.append({
                'timestamp': int(timestamps[i]),
                'nanoTime': int(nanotimes[i]),
                'type': sensor_type,
                'x': x_vals[i],
                'y': y_vals[i],
                'z': z_vals[i],
                'w': w_vals[i],
                'label': str(digit)
            })

    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df, duration

def generate_training_file(digit, file_idx, stats, base_timestamp=1761800000000):
    """生成一个训练文件（包含10个按键）"""

    current_timestamp = base_timestamp + np.random.randint(0, 50000000)
    current_nanotime = 8034000000000 + np.random.randint(0, 500000000000)

    keypresses = []

    # 生成10个按键（主要是目标数字，混合其他数字）
    digits = [digit] * 7 + list(np.random.choice([d for d in range(10) if d != digit], 3))
    np.random.shuffle(digits)

    for d in digits:
        keypress_df, duration = generate_keypress(d, stats, current_timestamp, current_nanotime)

        # 最后一个按键添加空label行（模拟按键间隔）
        # 增大间隔，确保重复数字能被正确分割（>500ms阈值）
        interval = np.random.uniform(600, 900)
        current_timestamp += duration + interval
        current_nanotime += (duration + interval) * 1000000

        # 在按键之间添加空label的传感器数据
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

        keypresses.append(keypress_df)
        keypresses.append(pd.DataFrame(inter_rows))

    return pd.concat(keypresses, ignore_index=True)

def generate_dataset(stats, samples_per_digit=200, output_dir='../sensor_data_synthetic/files'):
    """生成完整数据集"""
    print("\n" + "=" * 70)
    print("生成合成数据集...")
    print("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 清空现有文件
    for f in glob.glob(os.path.join(output_dir, '*.csv')):
        os.remove(f)

    for digit in range(10):
        if (digit, 'all') not in stats:
            print(f"跳过数字 {digit}（没有统计数据）")
            continue

        print(f"生成数字 {digit}...")

        num_files = samples_per_digit // 7  # 每个文件包含~7个目标数字

        for file_idx in range(num_files):
            df = generate_training_file(digit, file_idx, stats)

            timestamp = 1761800000000 + np.random.randint(0, 50000000)
            filename = f'synthetic_training_{timestamp}_{digit}_{file_idx}.csv'
            filepath = os.path.join(output_dir, filename)

            df.to_csv(filepath, index=False)

        print(f"  ✓ 已生成 {num_files} 个文件")

    print("\n" + "=" * 70)
    print("合成数据生成完成！")
    print(f"保存位置: {output_dir}")
    print("=" * 70)

def verify_data_quality(output_dir):
    """验证数据质量"""
    print("\n" + "=" * 70)
    print("验证数据质量...")
    print("=" * 70)

    digit_counts = {i: 0 for i in range(10)}
    abnormal_count = 0
    total_count = 0

    for csv_file in glob.glob(os.path.join(output_dir, '*.csv')):
        df = pd.read_csv(csv_file)

        for label in df['label'].unique():
            if pd.notna(label) and label != '':
                label_data = df[df['label'] == label]
                if len(label_data) > 0:
                    duration = label_data['timestamp'].max() - label_data['timestamp'].min()
                    digit = int(float(label))
                    digit_counts[digit] += 1
                    total_count += 1

                    if duration > 5000:
                        abnormal_count += 1

    print(f"\n总样本数: {total_count}")
    print(f"异常样本数: {abnormal_count} ({abnormal_count/total_count*100:.1f}% if total_count > 0 else 0)")
    print(f"\n每个数字的样本数:")
    for digit in range(10):
        print(f"  数字 {digit}: {digit_counts[digit]}")

    if abnormal_count == 0:
        print("\n✓ 数据质量：优秀！")
    else:
        print(f"\n⚠ 仍有 {abnormal_count} 个异常样本")

def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "合成数据生成器 v2")
    print("=" * 70)

    # 1. 分析正常数据
    stats = analyze_normal_data()

    if not stats:
        print("\n✗ 没有找到正常数据！")
        return

    # 2. 生成数据
    samples = int(input("\n每个数字生成多少个样本？(推荐200): ") or "200")
    generate_dataset(stats, samples_per_digit=samples)

    # 3. 验证
    verify_data_quality('../sensor_data_synthetic/files')

    print("\n" + "=" * 70)
    print("下一步：训练模型")
    print("=" * 70)
    print("\n/Users/east/CursorProjects/ml_code/iot/bin/python run_all.py \\")
    print("    --data_dir ../sensor_data_synthetic/files \\")
    print("    --model xgboost")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
