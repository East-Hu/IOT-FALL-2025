#!/bin/bash

echo "=== 清空传感器数据脚本 ==="
echo ""

# 设置ADB路径和设备ID
ADB="/Users/east/Library/Android/sdk/platform-tools/adb"
DEVICE="RFCXA1767LX"

echo "1️⃣  检查手机上的文件..."
echo "-----------------------------------"
$ADB -s $DEVICE shell ls -lh /sdcard/Android/data/com.example.iotproject/files/ 2>/dev/null | grep ".csv" || echo "   (没有找到CSV文件)"

echo ""
echo "2️⃣  删除手机上的所有CSV文件..."
echo "-----------------------------------"
$ADB -s $DEVICE shell "rm /sdcard/Android/data/com.example.iotproject/files/*.csv" 2>/dev/null
echo "   ✓ 删除命令已执行"

echo ""
echo "3️⃣  验证手机端删除结果..."
echo "-----------------------------------"
remaining=$($ADB -s $DEVICE shell "ls /sdcard/Android/data/com.example.iotproject/files/*.csv 2>/dev/null | wc -l")
remaining=$(echo $remaining | tr -d ' ')

if [ "$remaining" = "0" ] || [ -z "$remaining" ]; then
    echo "   ✅ 手机端CSV文件已全部清空"
else
    echo "   ⚠️  手机上还有 $remaining 个文件"
fi

echo ""
echo "4️⃣  清理电脑端数据..."
echo "-----------------------------------"
if [ -d "./sensor_data" ]; then
    rm -rf ./sensor_data
    echo "   ✓ sensor_data 目录已删除"
fi

if [ -d "./ml_code/processed_data" ]; then
    rm -rf ./ml_code/processed_data
    echo "   ✓ processed_data 目录已删除"
fi

if [ -f "./ml_code/features.csv" ]; then
    rm ./ml_code/features.csv
    echo "   ✓ features.csv 已删除"
fi

if [ -d "./ml_code/models" ]; then
    rm -rf ./ml_code/models
    echo "   ✓ models 目录已删除"
fi

echo "   ✅ 电脑端数据已清空"

echo ""
echo "5️⃣  重新创建必要的目录..."
echo "-----------------------------------"
mkdir -p sensor_data/files
echo "   ✓ sensor_data/files 目录已创建"

echo ""
echo "="
echo "="
echo "🎉 数据清空完成！"
echo "="
echo ""
echo "📱 现在可以："
echo "   1. 打开手机App"
echo "   2. 进入密码预测模式"
echo "   3. 开始收集新的数据"
echo ""
echo "💾 收集完成后运行："
echo "   ./export_data.sh    # 导出数据到电脑"
echo ""
