#!/usr/bin/env python3
"""
生成合成的高质量传感器数据
基于现有正常数据的统计特征生成
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def load_normal_data():
    """加载所有正常的按键数据（持续时间1-5秒）"""
    print("=" * 70)
    print("加载正常数据样本...")
    print("=" * 70)

    normal_samples = {}  # {digit: [list of dataframes]}

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
                        if digit not in normal_samples:
                            normal_samples[digit] = []
                        normal_samples[digit].append(label_data)

    # 统计
    print("\n每个数字的正常样本数：")
    for digit in range(10):
        count = len(normal_samples.get(digit, []))
        print(f"  数字 {digit}: {count} 个样本")

    return normal_samples


def get_sensor_column_names(df):
    """从原始数据推断传感器列名"""
    # 原始数据格式：timestamp, nanoTime, type, x, y, z, w, label
    # type 列包含：ACC, MAG, GYRO, ROT_VEC
    return {
        'timestamp': 'timestamp',
        'nanoTime': 'nanoTime',
        'type': 'type',
        'x': 'x',
        'y': 'y',
        'z': 'z',
        'w': 'w',
        'label': 'label'
    }

def extract_statistics(normal_samples):
    """提取每个数字的传感器数据统计特征"""
    print("\n" + "=" * 70)
    print("提取统计特征...")
    print("=" * 70)

    stats = {}

    for digit in range(10):
        if digit not in normal_samples or len(normal_samples[digit]) == 0:
            print(f"警告：数字 {digit} 没有正常样本！")
            continue

        samples = normal_samples[digit]

        # 收集所有传感器列的数据
        all_data = {
            'ACC_x': [], 'ACC_y': [], 'ACC_z': [],
            'GYRO_x': [], 'GYRO_y': [], 'GYRO_z': [],
            'MAG_x': [], 'MAG_y': [], 'MAG_z': [],
            'ROT_VEC_x': [], 'ROT_VEC_y': [], 'ROT_VEC_z': [], 'ROT_VEC_w': []
        }

        durations = []

        for sample in samples:
            # 持续时间
            duration = sample['timestamp'].max() - sample['timestamp'].min()
            durations.append(duration)

            # 传感器数据
            for col in all_data.keys():
                if col in sample.columns:
                    all_data[col].extend(sample[col].dropna().values)

        # 计算统计量
        stats[digit] = {
            'duration_mean': np.mean(durations),
            'duration_std': np.std(durations),
            'sensors': {}
        }

        for sensor, values in all_data.items():
            if len(values) > 0:
                stats[digit]['sensors'][sensor] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        print(f"  数字 {digit}: 持续时间 {stats[digit]['duration_mean']:.0f}±{stats[digit]['duration_std']:.0f}ms")

    return stats

def generate_synthetic_keypress(digit, stats, sampling_rate=50):
    """生成一个合成的按键事件"""

    if digit not in stats:
        raise ValueError(f"数字 {digit} 没有统计数据")

    digit_stats = stats[digit]

    # 生成持续时间（1-3秒，正态分布）
    duration = np.random.normal(
        digit_stats['duration_mean'],
        digit_stats['duration_std'] * 0.5  # 减小方差使数据更集中
    )
    duration = np.clip(duration, 1000, 3000)  # 限制在1-3秒

    # 计算样本点数
    num_samples = int(duration * sampling_rate / 1000)

    # 生成时间戳
    timestamps = np.linspace(0, duration, num_samples)

    # 生成传感器数据
    data = {
        'timestamp': timestamps,
        'label': str(digit)
    }

    for sensor, sensor_stats in digit_stats['sensors'].items():
        # 生成基础信号（正态分布 + 小幅度趋势 + 噪声）
        base_signal = np.random.normal(
            sensor_stats['mean'],
            sensor_stats['std'] * 0.8,  # 稍微减小方差
            num_samples
        )

        # 添加平滑的趋势（模拟手机移动）
        trend_freq = np.random.uniform(0.5, 2.0)  # Hz
        trend_amplitude = sensor_stats['std'] * 0.3
        trend = trend_amplitude * np.sin(2 * np.pi * trend_freq * timestamps / 1000)

        # 添加高频噪声
        noise = np.random.normal(0, sensor_stats['std'] * 0.1, num_samples)

        # 组合信号
        signal = base_signal + trend + noise

        # 限制范围
        signal = np.clip(signal, sensor_stats['min'], sensor_stats['max'])

        data[sensor] = signal

    return pd.DataFrame(data)

def generate_dataset(stats, num_samples_per_digit=200, output_dir='../sensor_data_synthetic/files'):
    """生成完整的合成数据集"""
    print("\n" + "=" * 70)
    print("生成合成数据集...")
    print("=" * 70)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 清空现有文件
    for f in glob.glob(os.path.join(output_dir, '*.csv')):
        os.remove(f)

    total_generated = 0

    # 为每个数字生成样本
    for digit in range(10):
        if digit not in stats:
            print(f"跳过数字 {digit}（没有统计数据）")
            continue

        print(f"\n生成数字 {digit} 的样本...")

        # 每10个按键一个文件（模拟真实收集）
        num_files = num_samples_per_digit // 10

        for file_idx in range(num_files):
            keypresses = []

            # 每个文件包含10个按键（随机数字，但确保当前数字出现）
            digits_in_file = [digit] + list(np.random.choice(10, 9))
            np.random.shuffle(digits_in_file)

            current_time = 0

            for d in digits_in_file:
                if d not in stats:
                    continue

                # 生成按键
                keypress = generate_synthetic_keypress(d, stats)

                # 调整时间戳（连续）
                keypress['timestamp'] = keypress['timestamp'] + current_time
                current_time = keypress['timestamp'].max() + np.random.uniform(100, 500)  # 按键间隔

                keypresses.append(keypress)

            # 合并所有按键
            if keypresses:
                combined = pd.concat(keypresses, ignore_index=True)

                # 保存文件
                timestamp = int(np.random.uniform(1761794960444, 1761854466124))
                filename = f'synthetic_training_{timestamp}_{digit}_{file_idx}.csv'
                filepath = os.path.join(output_dir, filename)
                combined.to_csv(filepath, index=False)

                total_generated += len(digits_in_file)

        print(f"  ✓ 已生成 {num_files} 个文件")

    print("\n" + "=" * 70)
    print(f"合成数据生成完成！")
    print(f"总共生成: {total_generated} 个按键事件")
    print(f"保存位置: {output_dir}")
    print("=" * 70)

    return output_dir

def verify_synthetic_data(output_dir):
    """验证生成的数据质量"""
    print("\n" + "=" * 70)
    print("验证数据质量...")
    print("=" * 70)

    total_keypresses = 0
    abnormal_count = 0
    digit_counts = {i: 0 for i in range(10)}

    for csv_file in glob.glob(os.path.join(output_dir, '*.csv')):
        df = pd.read_csv(csv_file)

        for label in df['label'].unique():
            if pd.notna(label) and label != '':
                label_data = df[df['label'] == label]

                if len(label_data) > 0:
                    duration = label_data['timestamp'].max() - label_data['timestamp'].min()
                    digit = int(float(label))

                    digit_counts[digit] += 1
                    total_keypresses += 1

                    if duration > 5000:
                        abnormal_count += 1

    print(f"\n总按键数: {total_keypresses}")
    print(f"异常按键数: {abnormal_count} ({abnormal_count/total_keypresses*100:.1f}%)")
    print(f"\n每个数字的样本数:")
    for digit in range(10):
        print(f"  数字 {digit}: {digit_counts[digit]}")

    min_count = min(digit_counts.values())
    max_count = max(digit_counts.values())
    balance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\n数据平衡比: {balance_ratio:.2f}")

    if abnormal_count == 0:
        print("\n✓ 数据质量：优秀（无异常）")
    elif abnormal_count < total_keypresses * 0.1:
        print("\n✓ 数据质量：良好（< 10%异常）")
    else:
        print("\n⚠ 数据质量：一般（≥ 10%异常）")

def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "合成数据生成器")
    print("=" * 70)

    # 1. 加载正常数据
    normal_samples = load_normal_data()

    if not normal_samples:
        print("\n✗ 错误：没有找到正常数据！")
        print("请确保 ../sensor_data/files/ 目录中有数据文件")
        return

    # 2. 提取统计特征
    stats = extract_statistics(normal_samples)

    # 3. 生成合成数据
    num_samples = int(input("\n每个数字生成多少个样本？(推荐200-300): ") or "200")
    output_dir = generate_dataset(stats, num_samples_per_digit=num_samples)

    # 4. 验证数据质量
    verify_synthetic_data(output_dir)

    print("\n" + "=" * 70)
    print("下一步：")
    print("=" * 70)
    print("\n1. 训练模型：")
    print("   cd ml_code")
    print("   /Users/east/CursorProjects/ml_code/iot/bin/python run_all.py \\")
    print("       --data_dir ../sensor_data_synthetic/files \\")
    print("       --model xgboost")

    print("\n2. 或者替换原始数据：")
    print("   mv ../sensor_data/files ../sensor_data_backup")
    print("   mv ../sensor_data_synthetic/files ../sensor_data/files")
    print("   cd ml_code")
    print("   /Users/east/CursorProjects/ml_code/iot/bin/python run_all.py --model xgboost")

    print("\n注意：")
    print("  - 合成数据基于你现有的22个正常文件的统计特征")
    print("  - 数据质量高，持续时间均为1-3秒")
    print("  - 适合用于课程项目的概念验证")
    print("  - 在报告中应说明使用了合成数据")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
