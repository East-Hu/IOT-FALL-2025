#!/bin/bash

echo "=== 导出传感器数据脚本 ==="
echo ""

# 设置ADB路径和设备ID
ADB="/Users/east/Library/Android/sdk/platform-tools/adb"
DEVICE="RFCXA1767LX"

echo "1️⃣  检查手机上的文件..."
echo "-----------------------------------"
file_count=$($ADB -s $DEVICE shell "ls /sdcard/Android/data/com.example.iotproject/files/*.csv 2>/dev/null | wc -l")
file_count=$(echo $file_count | tr -d ' ')

if [ "$file_count" = "0" ] || [ -z "$file_count" ]; then
    echo "   ❌ 手机上没有找到CSV文件"
    echo ""
    echo "   请先在手机App中收集数据："
    echo "   1. 打开App → 进入密码预测模式"
    echo "   2. 按数字键收集数据"
    echo "   3. 点击'完成并保存'"
    echo ""
    exit 1
else
    echo "   ✓ 找到 $file_count 个CSV文件"
fi

echo ""
echo "2️⃣  创建目标目录..."
echo "-----------------------------------"
mkdir -p sensor_data/files
echo "   ✓ sensor_data/files 目录已准备"

echo ""
echo "3️⃣  从手机导出数据到电脑..."
echo "-----------------------------------"
$ADB -s $DEVICE pull /sdcard/Android/data/com.example.iotproject/files/ ./sensor_data/

echo ""
echo "4️⃣  验证导出结果..."
echo "-----------------------------------"
local_count=$(ls ./sensor_data/files/*.csv 2>/dev/null | wc -l)
local_count=$(echo $local_count | tr -d ' ')

if [ "$local_count" = "$file_count" ]; then
    echo "   ✅ 成功导出 $local_count 个文件"
else
    echo "   ⚠️  导出数量不匹配："
    echo "      手机: $file_count 个"
    echo "      电脑: $local_count 个"
fi

echo ""
echo "5️⃣  数据统计..."
echo "-----------------------------------"
total_lines=0
for file in ./sensor_data/files/*.csv; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        total_lines=$((total_lines + lines))
    fi
done

echo "   总文件数: $local_count"
echo "   总数据行: $total_lines"
echo "   估计样本: $((total_lines / 25))"  # 假设每个按键约25行

echo ""
echo "="
echo "="
echo "🎉 数据导出完成！"
echo "="
echo ""
echo "📊 文件位置: ./sensor_data/files/"
echo ""
echo "🔍 查看数据:"
echo "   ls -lh sensor_data/files/          # 查看文件列表"
echo "   head -20 sensor_data/files/*.csv   # 查看数据内容"
echo ""
echo "🤖 训练模型:"
echo "   cd ml_code"
echo "   python run_all.py --data_dir ../sensor_data/files --model random_forest"
echo ""
