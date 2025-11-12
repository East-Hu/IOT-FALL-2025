#!/usr/bin/env python3
"""
生成测试数据集
为评估模型性能生成已知密码的测试样本
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# 导入生成器函数
sys.path.append(os.path.dirname(__file__))
from generate_synthetic_data_v2 import analyze_normal_data, generate_keypress

def generate_test_password(password, stats, output_dir, base_timestamp=1762000000000):
    """
    生成一个测试密码文件

    Args:
        password: 字符串，如 "12345"
        stats: 统计数据
        output_dir: 输出目录
        base_timestamp: 基础时间戳
    """
    current_timestamp = base_timestamp + np.random.randint(0, 10000000)
    current_nanotime = 8100000000000 + np.random.randint(0, 100000000000)

    keypresses = []

    for digit_char in password:
        digit = int(digit_char)

        # 生成按键
        keypress_df, duration = generate_keypress(digit, stats, current_timestamp, current_nanotime)
        keypresses.append(keypress_df)

        # 更新时间戳（添加按键间隔）
        # 增大间隔，确保重复数字能被正确分割（>500ms阈值）
        interval = np.random.uniform(600, 900)
        current_timestamp += duration + interval
        current_nanotime += (duration + interval) * 1000000

        # 添加空label的间隔数据
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

    # 合并所有按键
    combined = pd.concat(keypresses, ignore_index=True)

    # 保存文件
    filename = f'test_password_{password}_{int(current_timestamp)}.csv'
    filepath = os.path.join(output_dir, filename)
    combined.to_csv(filepath, index=False)

    return filepath

def main():
    print("\n" + "=" * 70)
    print(" " * 22 + "测试数据生成器")
    print("=" * 70)

    # 1. 加载统计数据
    print("\n加载训练数据的统计特征...")
    stats = analyze_normal_data()

    if not stats:
        print("✗ 错误：无法加载统计数据")
        return

    # 2. 创建测试数据目录
    test_dir = '../test_data'
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    # 清空现有测试数据
    for f in Path(test_dir).glob('*.csv'):
        f.unlink()

    # 3. 定义测试密码（15个）
    test_passwords = [
        "12345",  # 测试1: 顺序
        "54321",  # 测试2: 逆序
        "13579",  # 测试3: 奇数
        "24680",  # 测试4: 偶数
        "11111",  # 测试5: 重复
        "98765",  # 测试6: 大数逆序
        "02468",  # 测试7: 从0开始
        "19283",  # 测试8: 随机1
        "74650",  # 测试9: 随机2
        "36912",  # 测试10: 随机3
        # 以下5个留给用户自己测试
        "56789",  # 测试11: 高位顺序
        "00000",  # 测试12: 全0
        "99999",  # 测试13: 全9
        "50505",  # 测试14: 交替
        "84273",  # 测试15: 随机4
    ]

    print(f"\n生成 {len(test_passwords)} 个测试密码...")
    print("=" * 70)

    generated_files = []

    for i, password in enumerate(test_passwords, 1):
        filepath = generate_test_password(password, stats, test_dir)
        generated_files.append((password, filepath))

        mark = "✓" if i <= 10 else "○"
        note = "(用于自动测试)" if i <= 10 else "(留给用户测试)"
        print(f"{mark} 测试 {i:2d}: 密码 {password} {note}")

    print("\n" + "=" * 70)
    print("测试数据生成完成！")
    print("=" * 70)
    print(f"\n✓ 共生成 {len(generated_files)} 个测试文件")
    print(f"✓ 保存位置: {test_dir}")
    print(f"✓ 前10个用于自动测试和演示")
    print(f"✓ 后5个留给你手动测试")

    # 保存测试密码清单
    manifest_path = os.path.join(test_dir, 'test_manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("测试数据清单\n")
        f.write("=" * 70 + "\n\n")
        for i, (password, filepath) in enumerate(generated_files, 1):
            filename = os.path.basename(filepath)
            purpose = "自动测试" if i <= 10 else "手动测试"
            f.write(f"测试 {i:2d}: {password} - {filename} ({purpose})\n")

    print(f"\n✓ 测试清单已保存: {manifest_path}")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
