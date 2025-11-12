#!/bin/bash

# 演示用的自动化预测脚本
# 使用方法：./demo_predict.sh [实际密码]
# 例如：./demo_predict.sh 54321

echo ""
echo "=========================================="
echo "      密码预测演示自动化脚本"
echo "=========================================="
echo ""

# 检查是否提供了实际密码
if [ -z "$1" ]; then
    echo "❌ 错误：请提供实际密码"
    echo ""
    echo "使用方法："
    echo "  ./demo_predict.sh 54321"
    echo ""
    exit 1
fi

ACTUAL_PASSWORD=$1

# 1. 找到最新的模型
echo "1️⃣  查找最新的训练模型..."
echo "-------------------------------------------"
LATEST_MODEL=$(ls -t ml_code/models/*.pkl 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "   ❌ 错误：没有找到训练好的模型"
    echo ""
    echo "   请先训练模型："
    echo "   cd ml_code"
    echo "   python run_all.py --model random_forest"
    echo ""
    exit 1
fi

echo "   ✓ 找到模型: $(basename $LATEST_MODEL)"
echo ""

# 2. 找到最新的数据文件
echo "2️⃣  查找最新的测试数据..."
echo "-------------------------------------------"
LATEST_DATA=$(ls -t sensor_data/files/*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_DATA" ]; then
    echo "   ❌ 错误：没有找到数据文件"
    echo ""
    echo "   请先："
    echo "   1. 在手机App上输入密码"
    echo "   2. 运行 ./export_data.sh"
    echo ""
    exit 1
fi

echo "   ✓ 找到数据: $(basename $LATEST_DATA)"
echo ""

# 3. 显示文件信息
echo "3️⃣  数据文件信息..."
echo "-------------------------------------------"
DATA_SIZE=$(wc -l < "$LATEST_DATA")
DATA_DATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_DATA")
echo "   文件行数: $DATA_SIZE"
echo "   创建时间: $DATA_DATE"
echo ""

# 4. 确认信息
echo "4️⃣  准备预测..."
echo "-------------------------------------------"
echo "   模型: $(basename $LATEST_MODEL)"
echo "   数据: $(basename $LATEST_DATA)"
echo "   实际密码: $ACTUAL_PASSWORD"
echo ""

# 5. 等待用户确认
read -p "   按 Enter 开始预测，或按 Ctrl+C 取消..."
echo ""

# 6. 运行预测
echo "5️⃣  运行预测..."
echo "==========================================="
echo ""

cd ml_code

python 4_predict_password.py \
    --model "../$LATEST_MODEL" \
    --data "../$LATEST_DATA" \
    --actual "$ACTUAL_PASSWORD"

# 检查预测是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✅ 预测完成！"
    echo "==========================================="
    echo ""
else
    echo ""
    echo "==========================================="
    echo "❌ 预测过程出现错误"
    echo "==========================================="
    echo ""
    exit 1
fi
