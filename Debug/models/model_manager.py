import os
from datetime import datetime
import json
import shutil

class ModelManager:
    """简单的模型管理器"""
    
    def __init__(self, base_dir: str = "models"):
        """初始化
        
        Args:
            base_dir: 模型存储的基础目录
        """
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    def create_model_dir(self) -> str:
        """创建新的模型目录
        
        Returns:
            str: 模型目录路径
        """
        model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(model_dir)
        return model_dir
        
    def save_model(self, model_dir: str, model_path: str, meta_path: str, processor_path: str, 
                  chan_config: dict = None, train_info: dict = None) -> None:
        """保存模型文件到指定目录
        
        Args:
            model_dir: 模型目录
            model_path: 模型文件路径
            meta_path: 特征映射文件路径
            processor_path: 特征处理器文件路径
            chan_config: 缠论配置字典
            train_info: 训练信息，包含品种、周期等
        """
        # 复制文件到模型目录
        shutil.copy2(model_path, os.path.join(model_dir, "model.json"))
        shutil.copy2(meta_path, os.path.join(model_dir, "feature.meta"))
        shutil.copy2(processor_path, os.path.join(model_dir, "feature_processor.joblib"))
        
        # 合并配置信息到train_info
        if train_info is None:
            train_info = {}
        if chan_config:
            train_info['chan_config'] = chan_config
            
        # 保存训练信息
        if train_info:
            with open(os.path.join(model_dir, "train_info.json"), "w") as f:
                json.dump(train_info, f, indent=2) 