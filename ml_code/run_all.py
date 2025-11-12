"""
一键运行脚本
功能：自动完成数据预处理 → 特征提取 → 模型训练的完整流程

使用方法：
    python run_all.py --data_dir ./sensor_data --model random_forest
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import importlib.util
import pandas as pd

# 动态导入模块
spec1 = importlib.util.spec_from_file_location("data_preprocessing",
                                                os.path.join(os.path.dirname(__file__), "1_data_preprocessing.py"))
data_preprocessing = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(data_preprocessing)
SensorDataProcessor = data_preprocessing.SensorDataProcessor

spec2 = importlib.util.spec_from_file_location("feature_extraction",
                                                os.path.join(os.path.dirname(__file__), "2_feature_extraction.py"))
feature_extraction = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(feature_extraction)
FeatureExtractor = feature_extraction.FeatureExtractor

spec3 = importlib.util.spec_from_file_location("train_model",
                                                os.path.join(os.path.dirname(__file__), "3_train_model.py"))
train_model = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(train_model)
PasswordPredictor = train_model.PasswordPredictor

from sklearn.model_selection import train_test_split


def check_data_directory(data_dir):
    """检查数据目录是否存在且包含数据"""
    if not os.path.exists(data_dir):
        print(f"✗ 错误：数据目录不存在: {data_dir}")
        print(f"\n请先从手机导出数据：")
        print(f"  方法1（推荐）：回到项目根目录，运行 ./export_data.sh")
        print(f"  方法2（手动）：adb -s RFCXA1767LX pull /sdcard/Android/data/com.example.iotproject/files/ ../sensor_data/")
        return False

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if len(csv_files) == 0:
        print(f"✗ 错误：数据目录中没有 CSV 文件: {data_dir}")
        print(f"\n请确认：")
        print(f"  1. 已在手机App中收集数据")
        print(f"  2. 已运行 ../export_data.sh 或手动pull数据")
        print(f"  3. 数据路径正确: {data_dir}")
        return False

    print(f"✓ 找到 {len(csv_files)} 个 CSV 文件")
    return True


def main():
    parser = argparse.ArgumentParser(description='一键运行完整机器学习流程')
    parser.add_argument('--data_dir', type=str, default='../sensor_data/files',
                        help='原始数据目录（默认: ../sensor_data/files）')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'svm', 'logistic'],
                        help='模型类型')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 20 + "密码预测完整流程")
    print("=" * 70)

    # 检查数据
    if not check_data_directory(args.data_dir):
        sys.exit(1)

    # 定义输出路径
    processed_dir = './processed_data'
    features_file = './features.csv'
    models_dir = './models'

    print("\n" + "=" * 70)
    print("步骤 1/3: 数据预处理")
    print("=" * 70)

    # 步骤 1: 数据预处理
    processor = SensorDataProcessor(sampling_rate=50)
    segments = processor.process_all_files(args.data_dir)

    if len(segments) == 0:
        print("\n✗ 没有找到有效数据！")
        print("\n可能的原因：")
        print("  1. CSV 文件格式不正确")
        print("  2. 没有 label 列或 label 列为空")
        print("  3. 数据量太少（每次按键至少需要 0.2 秒数据）")
        sys.exit(1)

    processor.save_segments(segments, processed_dir)

    print("\n" + "=" * 70)
    print("步骤 2/3: 特征提取")
    print("=" * 70)

    # 步骤 2: 特征提取
    extractor = FeatureExtractor()
    features_df = extractor.process_all_keypresses(processed_dir)

    if features_df is None or len(features_df) == 0:
        print("\n✗ 特征提取失败！")
        sys.exit(1)

    features_df.to_csv(features_file, index=False)
    print(f"\n✓ 特征已保存: {features_file}")

    # 检查数据量
    min_samples = 50
    if len(features_df) < min_samples:
        print(f"\n⚠ 警告：样本数量较少（{len(features_df)} < {min_samples}）")
        print("  建议至少收集 100 个样本以获得更好的性能")
        response = input("\n是否继续训练？(y/n): ")
        if response.lower() != 'y':
            print("已取消训练")
            sys.exit(0)

    print("\n" + "=" * 70)
    print("步骤 3/3: 模型训练")
    print("=" * 70)

    # 步骤 3: 训练模型
    predictor = PasswordPredictor(model_type=args.model)

    # 加载特征
    X = features_df.drop('label', axis=1)
    y = features_df['label']

    # 确保标签是整数类型（对XGBoost很重要）
    y = y.apply(lambda x: int(float(x)) if pd.notna(x) else x)

    # 处理缺失值
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    # 保存特征名
    predictor.feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 创建并训练模型
    predictor.create_model()
    predictor.train(X_train, y_train)

    # 评估
    results = predictor.evaluate(X_test, y_test)

    # 特征重要性
    try:
        predictor.get_feature_importance()
    except:
        print("\n⚠ 该模型不支持特征重要性分析")

    # 保存模型
    model_path = predictor.save_model(models_dir)

    # 最终总结
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

    print(f"\n✓ 数据预处理: {len(segments)} 个按键事件")
    print(f"✓ 特征提取: {len(features_df)} 个样本, {len(X.columns)} 个特征")
    print(f"✓ 模型训练: 准确率 = {results['accuracy']*100:.2f}%")
    print(f"✓ 模型保存: {model_path}")

    print("\n下一步：")
    print("1. 收集新的密码数据进行测试")
    print("2. 使用以下命令进行预测：")
    print(f"\n   python 4_predict_password.py \\")
    print(f"       --model {model_path} \\")
    print(f"       --data <新密码数据.csv> \\")
    print(f"       --actual <实际密码>")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
