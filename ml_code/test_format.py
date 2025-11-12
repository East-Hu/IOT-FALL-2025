#!/usr/bin/env python3
"""
测试输出格式修复
"""

# 模拟预测结果
predictions = [
    {'digit': '4', 'confidence': 0.2213},
    {'digit': '4', 'confidence': 0.2575},
    {'digit': '6', 'confidence': 0.2275},
    {'digit': '8', 'confidence': 0.2051},
    {'digit': '5', 'confidence': 0.2099},
]

actual_password = "13579"

print("测试新的输出格式：")
print("=" * 60)

# 旧格式（错误）
old_format = ''.join([p['digit'] for p in predictions])
print(f"旧格式（错误）: {old_format}")

# 新格式（正确）
predicted_digits = [p['digit'] for p in predictions]
new_format = ' '.join(predicted_digits)
print(f"新格式（正确）: {new_format}")

# 实际密码
actual_digits = list(actual_password)
actual_format = ' '.join(actual_digits)
print(f"实际密码: {actual_format}")

print("\n逐位比较:")
for i, (p, a) in enumerate(zip(predicted_digits, actual_digits)):
    match = "✓" if p == a else "✗"
    print(f"  位置 {i+1}: 预测={p}, 实际={a} {match}")

print("\n" + "=" * 60)
print("✅ 格式修复成功！")
