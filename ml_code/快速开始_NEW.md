# 🚀 机器学习快速开始指南

这份文档帮助你快速开始使用机器学习模型进行密码预测。

---

## 📋 前置条件

### 1. 已收集数据
- ✅ 在手机App中收集了数据
- ✅ 运行了 `../export_data.sh` 导出数据
- ✅ 数据位于 `../sensor_data/files/` 目录

### 2. 已安装Python依赖
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
pip install xgboost  # 可选，准确率更高
```

---

## 🎯 一键运行（最简单）

```bash
cd /Users/east/AndroidStudioProjects/iotproject/ml_code
python run_all.py --model random_forest
```

就这么简单！脚本会自动完成数据预处理、特征提取和模型训练。

---

## 📊 查看结果

训练完成后会生成：
- `confusion_matrix_random_forest.png` - 混淆矩阵
- `feature_importance_random_forest.png` - 特征重要性
- `models/random_forest_*.pkl` - 训练好的模型

```bash
open confusion_matrix_random_forest.png
open feature_importance_random_forest.png
```

---

更多详细信息请查看 `README.md`
