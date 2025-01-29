import os
from datetime import datetime
import json
import shutil
from typing import Dict, Tuple, Optional, Any, Union
from autogluon.tabular import TabularPredictor
from Debug.models.feature_processor import FeatureProcessor
from ChanConfig import CChanConfig

class ModelManager:
    """模型管理器"""
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    def save_model(self, model_dir: str, feature_meta: Dict, processor, 
                  train_info: Dict) -> None:
        """保存模型和相关文件
        
        Args:
            model_dir: 模型目录
            model: 训练好的模型
            feature_meta: 特征映射
            processor: 特征处理器
            train_info: 训练信息
            chan_config: 缠论配置(字典或CChanConfig对象)
        """
        # 创建模型目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存特征映射
        with open(os.path.join(model_dir, "feature_meta.json"), "w") as f:
            json.dump(feature_meta, f, indent=2)
            
        # 保存特征处理器
        processor.save(os.path.join(model_dir, "feature_processor.joblib"))
            
        with open(os.path.join(model_dir, "train_info.json"), "w") as f:
            json.dump(train_info, f, indent=2)
            
    def load_model(self, model_dir: str) -> Tuple[Any, Dict, Any]:
        """加载模型和相关文件
        
        Args:
            model_dir: 模型目录
            
        Returns:
            model: 加载的模型
            feature_meta: 特征映射
            processor: 特征处理器
        """
        # 加载模型
        autogluon_path = os.path.join(model_dir, "autogluon")
        if not os.path.exists(autogluon_path):
            raise FileNotFoundError(f"AutoGluon 模型目录不存在: {autogluon_path}")
            
        try:
            model = TabularPredictor.load(autogluon_path)
        except Exception as e:
            raise Exception(f"加载 AutoGluon 模型失败: {str(e)}")
            
        # 加载特征映射
        feature_meta_path = os.path.join(model_dir, "feature_meta.json")
        if not os.path.exists(feature_meta_path):
            raise FileNotFoundError(f"特征映射文件不存在: {feature_meta_path}")
            
        with open(feature_meta_path, "r") as f:
            feature_meta = json.load(f)
            
        # 加载特征处理器
        processor_path = os.path.join(model_dir, "feature_processor.joblib")
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"特征处理器文件不存在: {processor_path}")
            
        processor = FeatureProcessor.load(processor_path)
        
        return model, feature_meta, processor
        
    def get_latest_model_dir(self) -> Optional[str]:
        """获取最新的模型目录"""
        if not os.path.exists(self.base_dir):
            return None
            
        # 过滤掉 __pycache__ 和其他不需要的目录
        model_dirs = [d for d in os.listdir(self.base_dir) 
                     if os.path.isdir(os.path.join(self.base_dir, d)) and 
                     not d.startswith('__') and  # 过滤掉 __pycache__ 等
                     not d.startswith('.')]      # 过滤掉 .git 等隐藏目录
        
        if not model_dirs:
            print(f"在 {self.base_dir} 中未找到有效的模型目录")
            return None
            
        # 按创建时间排序，获取最新的目录
        latest_dir = max(model_dirs, key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)))
        full_path = os.path.join(self.base_dir, latest_dir)
        print(f"找到最新模型目录: {full_path}")
        return full_path 