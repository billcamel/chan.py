import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import shutil
from PIL import Image
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split

class AutoGluonImageClassifier:
    """基于AutoGluon的图像分类器"""
    def __init__(self, 
                 model_dir: str = "image_models",
                 time_limit: int = 3600,
                 num_trials: int = 30):
        """初始化分类器
        
        Args:
            model_dir: 模型保存目录
            time_limit: 训练时间限制(秒)
            num_trials: 超参数搜索次数
        """
        self.model_dir = model_dir
        self.time_limit = time_limit
        self.num_trials = num_trials
        self.predictor = None
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def _prepare_data(self, data_dir: str) -> Dict:
        """准备训练数据
        
        Args:
            data_dir: 特征数据目录
            
        Returns:
            数据信息字典
        """
        # 加载标签数据
        labels = np.load(os.path.join(data_dir, "labels.npy"))
        
        # 加载特征数据
        features = np.load(os.path.join(data_dir, "features.npy"))
        
        # 加载特征元数据
        with open(os.path.join(data_dir, "feature_meta.json"), "r") as f:
            feature_meta = json.load(f)
            
        # 找到kline_image字段的索引
        kline_image_idx = None
        for idx, (name, _) in enumerate(feature_meta.items()):
            if name == 'kline_image':
                kline_image_idx = idx
                break
            
        if kline_image_idx is None:
            raise ValueError("在特征元数据中未找到 kline_image 字段")
            
        # 创建训练数据DataFrame
        data = []
        for idx, (label, feature) in enumerate(zip(labels, features)):
            # 获取图像名称
            image_name = feature[kline_image_idx]
            image_name = str(int(image_name))
            if not isinstance(image_name, str):
                print(f"警告: 索引 {idx} 的图像名称无效: {image_name}")
                continue
            
            image_path = os.path.join(data_dir, "images", str(image_name)+".png")
            if os.path.exists(image_path):
                data.append({
                    'image': image_path,
                    'label': int(label)  # 确保标签是整数
                })
            else:
                print(f"警告: 图像不存在: {image_path}")
            
        if not data:
            raise ValueError(f"未找到任何有效的图像数据在 {data_dir}")
        
        df = pd.DataFrame(data)
        
        # 打印数据统计
        print(f"\n数据统计:")
        print(f"总样本数: {len(df)}")
        print(f"标签分布:\n{df['label'].value_counts()}")
        
        # 划分训练集和验证集
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        return {
            'train_df': train_df,
            'val_df': val_df
        }
        
    def train(self, data_dir: str) -> Dict:
        """训练模型
        
        Args:
            data_dir: 特征数据目录
            
        Returns:
            训练结果信息
        """
        print("\n开始训练图像分类模型:")
        print(f"训练时间限制: {self.time_limit}秒")
        print(f"超参数搜索次数: {self.num_trials}")
        
        # 准备数据
        data_info = self._prepare_data(data_dir)
        print(f"训练样本数: {len(data_info['train_df'])}")
        print(f"验证样本数: {len(data_info['val_df'])}")
        
        # 创建预测器
        self.predictor = MultiModalPredictor(
            label='label',
            path=self.model_dir,
            problem_type='binary',
            eval_metric='roc_auc', 
            presets="high_quality"
        )

        # 指定使用 ResNet-50（通过 timm 库）
        hyperparameters = {
            'model.names': ['timm_image'],  # 使用 timm 图像模型
            'model.timm_image.checkpoint_name': 'resnet50.a1_in1k', # ResNet-50 预训练权重
            "optimization.max_epochs": 100,   # 训练轮次
            "optimization.patience":  15,
            "optimization.learning_rate": 2.0e-4,         # 学习率
            # 'env.per_gpu_batch_size': 32     # 根据显存调整批大小
        }
        
        # 训练模型
        self.predictor.fit(
            train_data=data_info['train_df'],
            tuning_data=data_info['val_df'],
            hyperparameters=hyperparameters,
            time_limit=self.time_limit,
        )
        print("训练完成,开始评估模型")
        # 评估模型
        val_metrics = self.predictor.evaluate(data_info['val_df'])
        
        results = {
            'val_metrics': val_metrics,
            'model_path': self.model_dir
        }
        
        print("\n模型评估完成:")
        print(f"验证指标: {val_metrics}")
        print(f"模型路径: {self.model_dir}")
        # 获取预测概率
        probs = self.predictor.predict_proba(data_info['train_df'])
        # 获取真实标签
        labels = data_info['train_df']['label']
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'label': labels,
            'prob': probs.iloc[:, 1]
        })
        # 保存到CSV文件
        results_df.to_csv('train_predictions.csv', index=False)
        print("预测结果已保存到 train_predictions.csv")
        return results
        
    def predict(self, image_path: str) -> float:
        """预测单个图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预测概率
        """
        if not os.path.exists(image_path):
            print(f"图像不存在: {image_path}")
            return 0
        if self.predictor is None:
            self.predictor = MultiModalPredictor.load(self.model_dir)
            
        # 创建测试数据
        test_data = pd.DataFrame([{'image': image_path}])
        
        # 预测
        probs = self.predictor.predict_proba(test_data)
        return probs.iloc[0, 1]  # 返回正类的概率
        
    @classmethod
    def load(cls, model_dir: str):
        """加载已训练的模型
        
        Args:
            model_dir: 模型目录
            
        Returns:
            分类器实例
        """
        classifier = cls(model_dir=model_dir)
        classifier.predictor = MultiModalPredictor.load(model_dir)
        return classifier

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练K线图像分类模型')
    parser.add_argument('--data', type=str, required=True,
                      help='特征数据目录')
    parser.add_argument('--time-limit', type=int, default=3600,
                      help='训练时间限制(秒)')
    parser.add_argument('--num-trials', type=int, default=30,
                      help='超参数搜索次数')
    parser.add_argument('--output', type=str, default='image_models',
                      help='模型输出目录')
    
    args = parser.parse_args()
    
    classifier = AutoGluonImageClassifier(
        model_dir=args.output,
        time_limit=args.time_limit,
        num_trials=args.num_trials
    )
    
    results = classifier.train(args.data) 
