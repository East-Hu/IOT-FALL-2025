#!/usr/bin/env python3
"""
自动运行测试并生成报告
"""

import pandas as pd
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

# 导入预测相关模块
import importlib.util
spec = importlib.util.spec_from_file_location("predict_password", "4_predict_password.py")
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
PasswordSequencePredictor = predict_module.PasswordSequencePredictor

def run_single_test(model_path, test_file, actual_password):
    """运行单个测试并返回结果"""

    # 创建预测器（静默模式）
    import io
    import contextlib

    # 抑制输出
    with contextlib.redirect_stdout(io.StringIO()):
        predictor = PasswordSequencePredictor(model_path)
        predictions, predicted_password = predictor.predict_password_sequence(test_file)

    # 计算准确率
    correct = 0
    total = len(actual_password)

    comparisons = []
    for i, (actual_digit, pred_dict) in enumerate(zip(actual_password, predictions)):
        predicted_digit = pred_dict['digit']
        is_correct = (predicted_digit == actual_digit)
        if is_correct:
            correct += 1

        comparisons.append({
            'position': i + 1,
            'actual': actual_digit,
            'predicted': predicted_digit,
            'correct': is_correct,
            'confidence': pred_dict['confidence']
        })

    accuracy = correct / total * 100

    return {
        'actual_password': actual_password,
        'predicted_password': predicted_password,
        'correct_digits': correct,
        'total_digits': total,
        'accuracy': accuracy,
        'comparisons': comparisons,
        'predictions': predictions
    }

def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "自动测试脚本")
    print("=" * 70)

    # 1. 找到最新的模型
    model_files = sorted(glob.glob('models/xgboost_*.pkl'))
    if not model_files:
        print("\n✗ 错误：未找到训练好的模型")
        print("请先运行: python run_all.py --model xgboost")
        return

    model_path = model_files[-1]
    print(f"\n使用模型: {os.path.basename(model_path)}")

    # 2. 定义测试密码（前10个）
    test_cases = [
        ("12345", "顺序数字"),
        ("54321", "逆序数字"),
        ("13579", "奇数序列"),
        ("24680", "偶数序列"),
        ("11111", "重复数字"),
        ("98765", "大数逆序"),
        ("02468", "从0开始"),
        ("19283", "随机组合1"),
        ("74650", "随机组合2"),
        ("36912", "随机组合3"),
    ]

    # 3. 找到对应的测试文件
    test_dir = '../test_data'
    test_files = []

    for password, description in test_cases:
        pattern = f'test_password_{password}_*.csv'
        matches = glob.glob(os.path.join(test_dir, pattern))
        if matches:
            test_files.append((password, description, matches[0]))
        else:
            print(f"⚠ 警告：未找到密码 {password} 的测试文件")

    if not test_files:
        print("\n✗ 错误：未找到任何测试文件")
        print("请先运行: python generate_test_data.py")
        return

    print(f"\n找到 {len(test_files)} 个测试文件")

    # 4. 运行测试
    print("\n" + "=" * 70)
    print("开始测试...")
    print("=" * 70)

    results = []

    for i, (password, description, test_file) in enumerate(test_files, 1):
        print(f"\n测试 {i}/10: {password} ({description})")
        print("-" * 70)

        try:
            result = run_single_test(model_path, test_file, password)
            results.append({
                'test_id': i,
                'password': password,
                'description': description,
                'file': os.path.basename(test_file),
                **result
            })

            # 显示结果
            print(f"实际密码: {result['actual_password']}")
            print(f"预测密码: {result['predicted_password']}")
            print(f"准确率: {result['accuracy']:.1f}% ({result['correct_digits']}/{result['total_digits']})")

            # 显示详细对比
            print("\n详细对比:")
            for comp in result['comparisons']:
                status = "✓" if comp['correct'] else "✗"
                print(f"  位置 {comp['position']}: {comp['actual']} → {comp['predicted']} {status} (置信度: {comp['confidence']:.1f}%)")

        except Exception as e:
            print(f"✗ 测试失败: {e}")
            results.append({
                'test_id': i,
                'password': password,
                'description': description,
                'file': os.path.basename(test_file),
                'error': str(e)
            })

    # 5. 生成汇总报告
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)

    successful_tests = [r for r in results if 'accuracy' in r]
    failed_tests = [r for r in results if 'error' in r]

    if successful_tests:
        total_accuracy = sum(r['accuracy'] for r in successful_tests) / len(successful_tests)
        perfect_tests = len([r for r in successful_tests if r['accuracy'] == 100])

        print(f"\n✓ 成功测试: {len(successful_tests)}/10")
        print(f"✓ 平均准确率: {total_accuracy:.2f}%")
        print(f"✓ 完全正确: {perfect_tests}/10")

        print("\n各测试准确率:")
        for r in successful_tests:
            bar_length = int(r['accuracy'] / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  测试 {r['test_id']:2d} ({r['password']}): {bar} {r['accuracy']:5.1f}%")

    if failed_tests:
        print(f"\n✗ 失败测试: {len(failed_tests)}/10")
        for r in failed_tests:
            print(f"  测试 {r['test_id']}: {r['password']} - {r['error']}")

    # 6. 保存详细报告
    output_dir = '../test_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'test_report_{timestamp}.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 22 + "测试结果报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用模型: {os.path.basename(model_path)}\n")
        f.write(f"测试数量: {len(results)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("总体结果\n")
        f.write("=" * 70 + "\n\n")

        if successful_tests:
            f.write(f"成功测试: {len(successful_tests)}/10\n")
            f.write(f"平均准确率: {total_accuracy:.2f}%\n")
            f.write(f"完全正确: {perfect_tests}/10\n\n")

        f.write("=" * 70 + "\n")
        f.write("详细结果\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"测试 {r['test_id']}: {r['password']} - {r['description']}\n")
            f.write("-" * 70 + "\n")

            if 'accuracy' in r:
                f.write(f"实际密码: {r['actual_password']}\n")
                f.write(f"预测密码: {r['predicted_password']}\n")
                f.write(f"准确率: {r['accuracy']:.1f}% ({r['correct_digits']}/{r['total_digits']})\n\n")

                f.write("详细对比:\n")
                for comp in r['comparisons']:
                    status = "✓ 正确" if comp['correct'] else "✗ 错误"
                    f.write(f"  位置 {comp['position']}: {comp['actual']} → {comp['predicted']} {status} (置信度: {comp['confidence']:.1f}%)\n")
            else:
                f.write(f"错误: {r['error']}\n")

            f.write("\n")

    print(f"\n✓ 详细报告已保存: {report_file}")

    # 7. 保存CSV格式的结果
    if successful_tests:
        csv_file = os.path.join(output_dir, f'test_results_{timestamp}.csv')
        csv_data = []

        for r in successful_tests:
            for comp in r['comparisons']:
                csv_data.append({
                    'test_id': r['test_id'],
                    'password': r['password'],
                    'description': r['description'],
                    'position': comp['position'],
                    'actual_digit': comp['actual'],
                    'predicted_digit': comp['predicted'],
                    'correct': comp['correct'],
                    'confidence': comp['confidence']
                })

        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"✓ CSV结果已保存: {csv_file}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
