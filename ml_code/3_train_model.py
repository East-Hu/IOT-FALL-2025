"""
模型训练模块
功能：训练多种机器学习模型预测密码数字

算法：
1. Random Forest (随机森林)
2. XGBoost
3. SVM (支持向量机)
4. Logistic Regression (逻辑回归)

使用方法：
    python 3_train_model.py --features ./features.csv --model random_forest
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

# 模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost 未安装，将跳过 XGBoost 模型")

import matplotlib.pyplot as plt
import seaborn as sns


class PasswordPredictor:
    """密码预测模型训练器"""

    def __init__(self, model_type='random_forest'):
        """
        初始化
        Args:
            model_type: 模型类型 (random_forest, xgboost, svm, logistic)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def create_model(self):
        """创建模型"""
        if self.model_type == 'random_forest':
            print("使用模型: Random Forest (随机森林)")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost 未安装！请运行: pip install xgboost")
            print("使用模型: XGBoost")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'svm':
            print("使用模型: SVM (支持向量机)")
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42,
                probability=True  # 启用概率预测
            )

        elif self.model_type == 'logistic':
            print("使用模型: Logistic Regression (逻辑回归)")
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )

        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

    def load_data(self, features_file):
        """
        加载特征数据
        Args:
            features_file: 特征文件路径
        Returns:
            X, y: 特征矩阵和标签
        """
        print(f"\n加载特征文件: {features_file}")
        df = pd.read_csv(features_file)

        print(f"  数据形状: {df.shape}")
        print(f"  特征数量: {len(df.columns) - 1}")
        print(f"  样本数量: {len(df)}")

        # 分离特征和标签
        X = df.drop('label', axis=1)
        y = df['label']

        # 处理缺失值
        X = X.fillna(0)

        # 处理无穷大
        X = X.replace([np.inf, -np.inf], 0)

        self.feature_names = X.columns.tolist()

        print(f"\n标签分布:")
        print(y.value_counts().sort_index())

        return X, y

    def train(self, X_train, y_train):
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练
        self.model.fit(X_train_scaled, y_train)

        print("✓ 训练完成！")

        # 交叉验证
        print("\n进行 5 折交叉验证...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)

        # 标准化
        X_test_scaled = self.scaler.transform(X_test)

        # 预测
        y_pred = self.model.predict(X_test_scaled)

        # 准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # 其他指标
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1 分数: {f1:.4f}")

        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, y_test.unique())

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'true_labels': y_test
        }

    def plot_confusion_matrix(self, cm, labels):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(labels),
                    yticklabels=sorted(labels))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {self.model_type}')

        # 保存图片
        output_file = f'confusion_matrix_{self.model_type}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ 混淆矩阵已保存: {output_file}")
        plt.close()

    def get_feature_importance(self, top_n=20):
        """获取特征重要性（仅 Random Forest 和 XGBoost）"""
        if self.model_type in ['random_forest', 'xgboost']:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            print(f"\nTop {top_n} 重要特征:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

            # 绘制特征重要性
            plt.figure(figsize=(12, 6))
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [self.feature_names[i] for i in indices],
                       rotation=45, ha='right')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
            plt.tight_layout()

            output_file = f'feature_importance_{self.model_type}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ 特征重要性图已保存: {output_file}")
            plt.close()

    def save_model(self, output_dir='./models'):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(output_dir,
                                  f'{self.model_type}_{timestamp}.pkl')

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, model_file)

        print(f"\n✓ 模型已保存: {model_file}")
        return model_file


def main():
    parser = argparse.ArgumentParser(description='训练密码预测模型')
    parser.add_argument('--features', type=str, default='./features.csv',
                        help='特征文件路径')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'svm', 'logistic'],
                        help='模型类型')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')

    args = parser.parse_args()

    print("=" * 60)
    print("密码预测模型训练")
    print("=" * 60)

    # 创建预测器
    predictor = PasswordPredictor(model_type=args.model)

    # 加载数据
    X, y = predictor.load_data(args.features)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 创建模型
    predictor.create_model()

    # 训练
    predictor.train(X_train, y_train)

    # 评估
    results = predictor.evaluate(X_test, y_test)

    # 特征重要性
    try:
        predictor.get_feature_importance()
    except:
        print("\n⚠ 该模型不支持特征重要性分析")

    # 保存模型
    predictor.save_model()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最终准确率: {results['accuracy']*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
