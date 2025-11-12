"""
特征提取模块
功能：从原始传感器数据提取统计特征和频域特征

使用方法：
    python 2_feature_extraction.py --input ./processed_data --output ./features.csv
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
import sys
from scipy import stats
try:
    from scipy.fft import fft
except ImportError:
    from numpy.fft import fft


class FeatureExtractor:
    """传感器数据特征提取器"""

    def __init__(self):
        self.sensor_types = ['ACC', 'GYRO', 'ROT_VEC', 'MAG']
        self.axes = ['x', 'y', 'z', 'w']

    def extract_time_domain_features(self, values):
        """
        提取时域特征
        Args:
            values: 一维数组
        Returns:
            dict: 特征字典
        """
        if len(values) == 0:
            return {}

        features = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'range': np.ptp(values),  # peak-to-peak
            'rms': np.sqrt(np.mean(values ** 2)),  # root mean square
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
        }

        # 过零率
        zero_crossings = np.where(np.diff(np.sign(values)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(values)

        return features

    def extract_frequency_domain_features(self, values, sampling_rate=50):
        """
        提取频域特征
        Args:
            values: 一维数组
            sampling_rate: 采样率
        Returns:
            dict: 特征字典
        """
        if len(values) < 4:
            return {}

        # FFT
        fft_values = fft(values)
        fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
        freqs = np.fft.fftfreq(len(values), 1/sampling_rate)[:len(fft_values)//2]

        features = {
            'fft_energy': np.sum(fft_magnitude ** 2),
            'fft_mean': np.mean(fft_magnitude),
            'fft_std': np.std(fft_magnitude),
            'fft_max': np.max(fft_magnitude),
        }

        # 主频率
        if len(fft_magnitude) > 0:
            dominant_freq_idx = np.argmax(fft_magnitude)
            features['dominant_frequency'] = freqs[dominant_freq_idx]
            features['dominant_magnitude'] = fft_magnitude[dominant_freq_idx]

        # 频谱质心
        if np.sum(fft_magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)

        return features

    def extract_sensor_features(self, sensor_data, sensor_type):
        """
        提取单个传感器的所有特征
        Args:
            sensor_data: DataFrame，包含一个传感器类型的数据
            sensor_type: 传感器类型 (ACC, GYRO, etc.)
        Returns:
            dict: 特征字典
        """
        features = {}

        for axis in self.axes:
            if axis in sensor_data.columns:
                values = sensor_data[axis].values

                # 跳过空值或全 NaN
                if len(values) == 0 or np.all(np.isnan(values)):
                    continue

                # 移除 NaN
                values = values[~np.isnan(values)]

                if len(values) == 0:
                    continue

                # 时域特征
                time_features = self.extract_time_domain_features(values)
                for feat_name, feat_value in time_features.items():
                    features[f'{sensor_type}_{axis}_{feat_name}'] = feat_value

                # 频域特征
                freq_features = self.extract_frequency_domain_features(values)
                for feat_name, feat_value in freq_features.items():
                    features[f'{sensor_type}_{axis}_{feat_name}'] = feat_value

        return features

    def extract_keypress_features(self, keypress_data):
        """
        提取一次按键的所有特征
        Args:
            keypress_data: DataFrame，包含一次按键的所有传感器数据
        Returns:
            dict: 特征字典
        """
        all_features = {}

        # 为每种传感器提取特征
        for sensor_type in self.sensor_types:
            sensor_data = keypress_data[keypress_data['type'] == sensor_type]

            if len(sensor_data) > 0:
                sensor_features = self.extract_sensor_features(sensor_data, sensor_type)
                all_features.update(sensor_features)

        # 添加时间相关特征
        if 'timestamp' in keypress_data.columns:
            timestamps = keypress_data['timestamp'].values
            all_features['duration_ms'] = timestamps[-1] - timestamps[0]

        all_features['total_samples'] = len(keypress_data)

        return all_features

    def process_all_keypresses(self, data_dir):
        """
        处理所有按键数据文件
        Args:
            data_dir: 包含按键数据的目录
        Returns:
            DataFrame: 特征矩阵
        """
        keypress_files = sorted(glob.glob(os.path.join(data_dir, 'keypress_*.csv')))

        if len(keypress_files) == 0:
            print("✗ 没有找到按键数据文件！")
            return None

        print(f"找到 {len(keypress_files)} 个按键文件\n")

        features_list = []
        labels = []

        for i, filepath in enumerate(keypress_files):
            # 从文件名提取标签
            filename = os.path.basename(filepath)
            label_str = filename.split('_label_')[1].replace('.csv', '')

            # 转换为整数（处理浮点数格式）
            try:
                label = int(float(label_str))
            except:
                label = label_str

            # 读取数据
            df = pd.read_csv(filepath)

            # 提取特征
            features = self.extract_keypress_features(df)

            if len(features) > 0:
                features_list.append(features)
                labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(keypress_files)} 个文件...")

        print(f"\n✓ 提取完成！共 {len(features_list)} 个样本")

        # 转换为 DataFrame
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels

        # 显示特征数量
        print(f"✓ 特征数量: {len(features_df.columns) - 1}")
        print(f"✓ 样本数量: {len(features_df)}")

        # 显示每个标签的数量
        print("\n每个数字的样本数:")
        print(features_df['label'].value_counts().sort_index())

        return features_df


def main():
    parser = argparse.ArgumentParser(description='提取传感器特征')
    parser.add_argument('--input', type=str, default='./processed_data',
                        help='预处理后的数据目录')
    parser.add_argument('--output', type=str, default='./features.csv',
                        help='输出特征文件')

    args = parser.parse_args()

    print("=" * 60)
    print("特征提取")
    print("=" * 60)

    # 创建特征提取器
    extractor = FeatureExtractor()

    # 提取特征
    features_df = extractor.process_all_keypresses(args.input)

    if features_df is None or len(features_df) == 0:
        print("\n✗ 特征提取失败！")
        return

    # 保存特征
    features_df.to_csv(args.output, index=False)
    print(f"\n✓ 特征已保存到: {args.output}")

    print("\n" + "=" * 60)
    print("特征提取完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
