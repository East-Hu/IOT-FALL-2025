# 密码预测机器学习代码 - 使用指南

## 📂 文件说明

本文件夹包含完整的密码预测机器学习代码：

```
ml_code/
├── 1_data_preprocessing.py      # 步骤1：数据预处理
├── 2_feature_extraction.py      # 步骤2：特征提取
├── 3_train_model.py             # 步骤3：模型训练
├── 4_predict_password.py        # 步骤4：密码预测
├── run_all.py                   # 一键运行所有步骤
├── README_CN.md                 # 本文件（中文说明）
├── README.md                    # 英文详细文档
└── 快速开始.md                   # 快速入门教程
```

---

## ⚡ 快速开始（3步）

### 第1步：安装Python依赖包

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

可选（更高准确率）：
```bash
pip install xgboost
```

### 第2步：从手机导出数据

```bash
# 导出到 sensor_data 文件夹
adb pull /data/data/com.example.iotproject/files/ ./sensor_data/
```

### 第3步：运行模型训练

```bash
cd ml_code
python run_all.py --data_dir ../sensor_data --model random_forest
```

**完成！** 程序会自动完成：
1. ✅ 数据预处理（分割按键）
2. ✅ 特征提取（提取200+特征）
3. ✅ 模型训练（训练分类器）
4. ✅ 性能评估（显示准确率）

---

## 🎯 算法说明

### 使用的4种机器学习算法

| 算法 | 准确率 | 速度 | 推荐度 | 说明 |
|-----|--------|------|--------|------|
| **Random Forest** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **推荐！**随机森林，100棵决策树投票 |
| **XGBoost** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 梯度提升树，准确率最高 |
| **SVM** | ⭐⭐⭐ | ⭐ | ⭐⭐ | 支持向量机，适合小数据集 |
| **Logistic Regression** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 逻辑回归，基线模型 |

### 默认算法：Random Forest（随机森林）

**工作原理**：
1. 建立100棵决策树
2. 每棵树用不同的数据训练
3. 预测时所有树投票，多数获胜

**为什么选它**：
- ✅ 准确率高（60-75%）
- ✅ 训练快（几秒钟）
- ✅ 不容易过拟合
- ✅ 可以看哪些特征重要

**超参数设置**：
```python
RandomForestClassifier(
    n_estimators=100,      # 100棵树
    max_depth=20,          # 最大深度20
    min_samples_split=5,   # 分裂最少需要5个样本
    random_state=42        # 随机种子
)
```

---

## 📊 特征提取说明

### 提取的特征数量：约200-300个

#### 从4种传感器提取特征：
1. **加速度计 (ACC)** - 检测手机移动
2. **陀螺仪 (GYRO)** - 检测旋转
3. **旋转矢量 (ROT_VEC)** - 设备方向
4. **磁力计 (MAG)** - 磁场强度

#### 每个传感器轴提取18种特征：

**时域特征（11个）**：
- 均值、标准差、最大值、最小值、中位数
- 范围、均方根、偏度、峰度
- 四分位数、过零率

**频域特征（7个）**：
- FFT能量、频谱均值、频谱标准差
- 主频率、主频幅值、频谱质心

**计算方式**：
```
4种传感器 × 3-4轴 × 18特征 = 200+特征
```

---

## 📖 详细使用方法

### 方式1：一键运行（推荐初学者）

```bash
python run_all.py --data_dir ../sensor_data --model random_forest
```

这个命令会自动执行：
1. 数据预处理
2. 特征提取
3. 模型训练
4. 性能评估

### 方式2：分步执行（推荐深入学习）

#### 步骤1：数据预处理

```bash
python 1_data_preprocessing.py \
    --data_dir ../sensor_data \
    --output ./processed_data
```

**功能**：读取CSV文件，按label分割成单个按键事件

**输入**：`sensor_data/password_training_*.csv`

**输出**：
- `processed_data/keypress_0000_label_5.csv` - 第1次按键（数字5）
- `processed_data/keypress_0001_label_3.csv` - 第2次按键（数字3）
- `processed_data/label_statistics.csv` - 统计信息

---

#### 步骤2：特征提取

```bash
python 2_feature_extraction.py \
    --input ./processed_data \
    --output ./features.csv
```

**功能**：从每次按键的传感器数据中提取统计特征

**输入**：`processed_data/keypress_*.csv`

**输出**：`features.csv` - 特征矩阵（每行是一次按键，每列是一个特征）

---

#### 步骤3：模型训练

```bash
# Random Forest（推荐）
python 3_train_model.py \
    --features ./features.csv \
    --model random_forest

# XGBoost（更高准确率）
python 3_train_model.py \
    --features ./features.csv \
    --model xgboost

# SVM
python 3_train_model.py \
    --features ./features.csv \
    --model svm

# Logistic Regression（基线）
python 3_train_model.py \
    --features ./features.csv \
    --model logistic
```

**功能**：训练机器学习模型预测数字（0-9）

**输入**：`features.csv`

**输出**：
- `models/random_forest_20250115_123456.pkl` - 训练好的模型
- `confusion_matrix_random_forest.png` - 混淆矩阵图
- `feature_importance_random_forest.png` - 特征重要性图

**评估指标**：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵

---

#### 步骤4：预测密码

```bash
python 4_predict_password.py \
    --model ./models/random_forest_20250115_123456.pkl \
    --data ./new_password_data.csv \
    --actual 12345  # 可选，用于评估
```

**功能**：使用训练好的模型预测新输入的密码

**输入**：
- 训练好的模型文件
- 新的密码数据CSV文件
- （可选）实际密码，用于对比评估

**输出**：
- 预测的密码序列
- 每个数字的置信度
- 如果提供了实际密码，显示准确率

---

## 📈 预期性能

### 准确率与数据量关系

| 每个数字样本数 | 总样本数 | Random Forest | XGBoost |
|--------------|---------|--------------|---------|
| 10次         | 100     | 30-40%       | 35-45%  |
| 50次         | 500     | 50-65%       | 55-70%  |
| 100次        | 1000    | 60-75%       | 65-80%  |
| 200次+       | 2000+   | 70-85%       | 75-90%  |

**注意**：
- 准确率 > 10%（随机猜测）就说明模型有效
- 准确率 > 50% 说明存在严重的隐私泄露风险
- 准确率 > 70% 说明可以实际破解密码

---

## 🛠️ 常见问题解决

### Q1: 提示"没有找到有效数据"

**原因**：CSV文件中没有label列或label为空

**解决方法**：
1. 确保使用APP的"密码预测模式"收集数据
2. 检查是否点击了数字按钮（会自动标注label）
3. 查看CSV文件确认有label列

### Q2: 准确率只有10%左右（随机水平）

**原因**：数据量太少或数据质量差

**解决方法**：
1. 至少收集100个样本（每个数字10次）
2. 确保按键自然、稳定
3. 增加数据量到500+样本

### Q3: 提示缺少Python包

**解决方法**：
```bash
# 安装基础包
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib

# 安装XGBoost（可选）
pip install xgboost
```

### Q4: 内存不足

**解决方法**：
1. 减少数据量
2. 使用更简单的模型（logistic）
3. 关闭其他程序

### Q5: 想了解更多技术细节

**解决方法**：
阅读 `README.md`（英文详细文档）或 `快速开始.md`

---

## 💡 提高准确率的方法

### 方法1：收集更多数据 ⭐⭐⭐⭐⭐（最重要！）

```
100样本  → 30-40% 准确率
500样本  → 50-65% 准确率
1000样本 → 60-75% 准确率
2000样本 → 70-85% 准确率
```

### 方法2：提高数据质量 ⭐⭐⭐⭐

- ✅ 按键节奏自然稳定
- ✅ 持握方式一致
- ✅ 避免太快或太慢
- ✅ 在相似环境下收集

### 方法3：尝试不同算法 ⭐⭐⭐

```bash
# 对比所有算法
python run_all.py --model random_forest
python run_all.py --model xgboost
python run_all.py --model svm
python run_all.py --model logistic
```

XGBoost通常比Random Forest高5-10%

### 方法4：调整超参数 ⭐⭐

修改 `3_train_model.py` 中的参数：

```python
# Random Forest
RandomForestClassifier(
    n_estimators=200,      # 增加树的数量
    max_depth=30,          # 增加深度
    # ...
)

# XGBoost
XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,    # 降低学习率
    # ...
)
```

---

## 🎓 实验建议

### 实验1：基线测试

**数据需求**：100个样本（每个数字10次）

**目标**：建立性能基线

```bash
python run_all.py --data_dir ../sensor_data --model random_forest
```

**预期结果**：30-40%准确率

---

### 实验2：增加数据量

**数据需求**：500个样本（每个数字50次）

**目标**：验证数据量的影响

```bash
# 收集更多数据后重新训练
python run_all.py --data_dir ../sensor_data --model random_forest
```

**预期结果**：50-65%准确率

---

### 实验3：算法对比

**数据需求**：500+样本

**目标**：找到最佳算法

```bash
python run_all.py --model random_forest
python run_all.py --model xgboost
python run_all.py --model svm
python run_all.py --model logistic
```

**对比指标**：
- 准确率
- 训练时间
- 预测速度

---

### 实验4：跨用户测试（可选）

**目标**：测试模型泛化能力（隐私风险评估）

**步骤**：
1. 用自己的数据训练模型
2. 让朋友输入密码
3. 测试能否预测朋友的密码

**意义**：
- 能预测 → 隐私风险大（传感器泄露通用模式）
- 不能预测 → 风险相对小（只对个人有效）

---

## 📝 项目报告建议

### 在报告中应包括的内容：

#### 1. Methodology（方法论）部分：

**数据预处理**：
- 如何分割按键事件
- 数据清洗方法

**特征提取**：
- 提取了哪些特征（时域+频域）
- 为什么选择这些特征
- 特征总数

**模型选择**：
- 选择的算法及理由
- 超参数设置
- 为什么Random Forest适合这个任务

**训练策略**：
- 训练集/测试集划分（80/20）
- 标准化方法
- 交叉验证

#### 2. Evaluation（评估）部分：

**数据集统计**：
- 总样本数
- 每个数字的分布
- 数据收集场景

**性能指标**：
- 准确率、精确率、召回率、F1分数
- 混淆矩阵分析
- 哪些数字容易混淆

**对比实验**：
- 不同算法的对比
- 不同数据量的影响

**特征分析**：
- 哪些传感器最重要
- 哪些特征贡献最大

#### 3. Discussion（讨论）部分：

**为什么能预测密码**：
- 每个数字的位置不同
- 按键时手机倾斜角度不同
- 传感器能捕捉这些微小差异

**隐私风险分析**：
- 这种攻击的可行性
- 潜在的威胁场景
- 防护措施建议

**局限性**：
- 需要大量训练数据
- 对个人依赖性强
- 实际攻击困难

**改进方向**：
- 深度学习方法（LSTM/CNN）
- 更多特征工程
- 集成学习

---

## 📦 输出文件说明

运行后会生成以下文件和文件夹：

```
ml_code/
├── processed_data/              # 预处理后的数据
│   ├── keypress_0000_label_5.csv
│   ├── keypress_0001_label_3.csv
│   ├── ...
│   └── label_statistics.csv
│
├── features.csv                 # 特征矩阵
│
├── models/                      # 训练好的模型
│   ├── random_forest_20250115_123456.pkl
│   ├── xgboost_20250115_123457.pkl
│   └── ...
│
├── confusion_matrix_random_forest.png      # 混淆矩阵图
└── feature_importance_random_forest.png    # 特征重要性图
```

---

## 🚀 下一步行动

### 立即开始：

1. ✅ **安装依赖**
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
   ```

2. ✅ **收集测试数据**（每个数字10次）
   - 打开APP → 进入密码预测模式
   - 随机按数字，重复10-20组

3. ✅ **导出数据**
   ```bash
   adb pull /data/data/com.example.iotproject/files/ ./sensor_data/
   ```

4. ✅ **运行第一个模型**
   ```bash
   cd ml_code
   python run_all.py --data_dir ../sensor_data
   ```

5. ✅ **查看结果**
   - 查看终端输出的准确率
   - 打开 `confusion_matrix_random_forest.png`
   - 分析哪些数字容易混淆

### 后续优化：

6. ✅ **收集更多数据**（500-1000样本）
7. ✅ **尝试XGBoost**（更高准确率）
8. ✅ **分析特征重要性**
9. ✅ **写实验报告**

---

## 📞 技术支持

### 遇到问题？

1. 查看本文档的"常见问题解决"部分
2. 阅读 `快速开始.md` 获取更详细的教程
3. 检查Python包是否都已安装
4. 确认数据格式是否正确

### 想深入了解？

- 查看 `README.md`（英文详细技术文档）
- 阅读代码中的注释
- 参考 `Part2_模型设计建议.md`

---

## 🎉 祝你实验成功！

**记住**：
- 数据质量 > 数据数量
- 耐心收集数据是成功的关键
- 从简单模型开始，逐步优化

有任何问题随时查看文档或询问！
