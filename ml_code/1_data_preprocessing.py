"""
数据预处理模块
功能：读取 CSV 文件，分割按键事件，准备训练数据

使用方法：
    python 1_data_preprocessing.py --data_dir ./sensor_data --output ./processed_data
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import sys


class SensorDataProcessor:
    """传感器数据预处理器"""

    def __init__(self, sampling_rate=50):
        """
        初始化
        Args:
            sampling_rate: 采样率（Hz），默认 50Hz
        """
        self.sampling_rate = sampling_rate

    def load_csv_file(self, filepath):
        """
        加载单个 CSV 文件
        Args:
            filepath: CSV 文件路径
        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ 加载文件: {os.path.basename(filepath)} - {len(df)} 行")
            return df
        except Exception as e:
            print(f"✗ 加载失败 {filepath}: {e}")
            return None

    def segment_by_label(self, df, time_gap_threshold=500):
        """
        按 label 分割数据，支持检测相同label的重复按键

        改进：即使label相同，如果时间间隔超过阈值，也会分割成不同的按键
        这样可以正确识别重复数字，如1111、9999等

        Args:
            df: 包含 label 列的 DataFrame
            time_gap_threshold: 时间间隔阈值（毫秒），默认500ms
        Returns:
            list of dict: [{'label': '5', 'data': DataFrame}, ...]
        """
        segments = []

        # 移除空标签的行
        df = df[df['label'].notna() & (df['label'] != '')].copy()

        if len(df) == 0:
            return segments

        # 按时间戳排序
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 第一步：找到 label 变化的点
        df['label_changed'] = df['label'] != df['label'].shift(1)

        # 第二步：在相同label内，检测时间间隔
        # 计算时间差（毫秒）
        df['time_diff'] = df['timestamp'].diff()

        # 如果时间间隔超过阈值，也认为是新的按键
        df['time_gap'] = df['time_diff'] > time_gap_threshold

        # 综合判断：label变化 或 时间间隔过大
        df['new_segment'] = df['label_changed'] | df['time_gap']
        df['segment_id'] = df['new_segment'].cumsum()

        # 按 segment_id 分组
        for segment_id, group in df.groupby('segment_id'):
            label = group['label'].iloc[0]
            # 只保留有足够数据的片段（至少 0.2 秒的数据）
            min_samples = int(0.2 * self.sampling_rate * 4)  # 4 种传感器
            if len(group) >= min_samples:
                segments.append({
                    'label': label,
                    'data': group.drop(['label_changed', 'time_diff', 'time_gap', 'new_segment', 'segment_id'], axis=1)
                })

        return segments

    def process_all_files(self, data_dir):
        """
        处理目录中的所有 CSV 文件
        Args:
            data_dir: 数据目录
        Returns:
            list of segments
        """
        all_segments = []

        # 查找所有 CSV 文件
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
        print(f"\n找到 {len(csv_files)} 个 CSV 文件\n")

        for filepath in csv_files:
            df = self.load_csv_file(filepath)
            if df is not None:
                segments = self.segment_by_label(df)
                all_segments.extend(segments)
                print(f"  → 提取了 {len(segments)} 个按键事件")

        print(f"\n总计提取了 {len(all_segments)} 个按键事件")
        return all_segments

    def save_segments(self, segments, output_dir):
        """
        保存分割后的数据
        Args:
            segments: 分割后的数据列表
            output_dir: 输出目录
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 保存每个片段
        for i, segment in enumerate(segments):
            filename = f"keypress_{i:04d}_label_{segment['label']}.csv"
            filepath = os.path.join(output_dir, filename)
            segment['data'].to_csv(filepath, index=False)

        print(f"\n✓ 保存了 {len(segments)} 个文件到 {output_dir}")

        # 保存统计信息
        labels = [s['label'] for s in segments]
        stats_df = pd.DataFrame({'label': labels})
        stats = stats_df['label'].value_counts().sort_index()

        print("\n每个数字的按键次数:")
        print(stats)

        # 保存统计到文件
        stats.to_csv(os.path.join(output_dir, 'label_statistics.csv'))


def main():
    parser = argparse.ArgumentParser(description='预处理传感器数据')
    parser.add_argument('--data_dir', type=str, default='./sensor_data',
                        help='原始 CSV 文件目录')
    parser.add_argument('--output', type=str, default='./processed_data',
                        help='输出目录')

    args = parser.parse_args()

    print("=" * 60)
    print("传感器数据预处理")
    print("=" * 60)

    # 创建处理器
    processor = SensorDataProcessor(sampling_rate=50)

    # 处理所有文件
    segments = processor.process_all_files(args.data_dir)

    if len(segments) == 0:
        print("\n✗ 没有找到有效数据！请检查数据目录。")
        return

    # 保存结果
    processor.save_segments(segments, args.output)

    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
