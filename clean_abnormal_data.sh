#!/bin/bash

echo "========================================================================"
echo "                    清理异常数据脚本"
echo "========================================================================"
echo ""

# 备份原始数据
echo "1️⃣  创建备份..."
echo "------------------------------------------------------------------------"
BACKUP_DIR="./sensor_data_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r ./sensor_data/files "$BACKUP_DIR/"
echo "   ✓ 数据已备份到: $BACKUP_DIR"
echo ""

# 统计原始文件数
ORIGINAL_COUNT=$(ls ./sensor_data/files/*.csv 2>/dev/null | wc -l)
echo "   原始CSV文件数: $ORIGINAL_COUNT"
echo ""

# 删除异常文件（持续时间超过5秒的按键）
echo "2️⃣  查找并删除异常文件..."
echo "------------------------------------------------------------------------"

python3 -c "
import pandas as pd
import glob
import os

abnormal_files = set()

for csv_file in sorted(glob.glob('./sensor_data/files/*.csv')):
    df = pd.read_csv(csv_file)
    filename = csv_file.split('/')[-1]

    has_abnormal = False
    for label in df['label'].unique():
        if pd.notna(label) and label != '':
            label_data = df[df['label'] == label]
            if len(label_data) > 0:
                duration = label_data['timestamp'].max() - label_data['timestamp'].min()
                if duration > 5000:  # 超过5秒
                    has_abnormal = True
                    break

    if has_abnormal:
        abnormal_files.add(csv_file)
        print(f'删除: {filename}')

# 删除文件
for f in abnormal_files:
    os.remove(f)

print(f'\n共删除 {len(abnormal_files)} 个异常文件')
"

echo ""

# 统计剩余文件数
REMAINING_COUNT=$(ls ./sensor_data/files/*.csv 2>/dev/null | wc -l)
DELETED_COUNT=$((ORIGINAL_COUNT - REMAINING_COUNT))

echo "3️⃣  清理结果"
echo "------------------------------------------------------------------------"
echo "   原始文件数: $ORIGINAL_COUNT"
echo "   删除文件数: $DELETED_COUNT"
echo "   剩余文件数: $REMAINING_COUNT"
echo ""

if [ $REMAINING_COUNT -lt 30 ]; then
    echo "⚠️  警告：剩余文件数较少（< 30），可能需要重新收集数据"
    echo ""
else
    echo "✅ 清理完成！剩余数据充足"
    echo ""
fi

echo "4️⃣  下一步"
echo "------------------------------------------------------------------------"
echo "   重新训练模型："
echo "   cd ml_code"
echo "   /Users/east/CursorProjects/ml_code/iot/bin/python run_all.py --model xgboost"
echo ""
echo "========================================================================"
