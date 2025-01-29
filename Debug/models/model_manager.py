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
        model = TabularPredictor.load(os.path.join(model_dir, "model"))
        
        # 加载特征映射
        with open(os.path.join(model_dir, "feature_meta.json"), "r") as f:
            feature_meta = json.load(f)
            
        # 加载特征处理器
        processor = FeatureProcessor.load(os.path.join(model_dir, "feature_processor.joblib"))
        
        return model, feature_meta, processor
        
    def get_latest_model_dir(self) -> Optional[str]:
        """获取最新的模型目录"""
        if not os.path.exists(self.base_dir):
            return None
            
        model_dirs = [d for d in os.listdir(self.base_dir) 
                     if os.path.isdir(os.path.join(self.base_dir, d))]
        
        if not model_dirs:
            return None
            
        latest_dir = max(model_dirs, key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)))
        return os.path.join(self.base_dir, latest_dir) 