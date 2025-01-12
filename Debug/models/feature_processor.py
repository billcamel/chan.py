"""特征处理器相关类"""
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import RobustScaler
import joblib

class FeatureProcessor:
    """特征归一化处理器"""
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """训练特征处理器"""
        self.feature_names = feature_names
        self.scaler = RobustScaler()
        self.scaler.fit(X)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换特征矩阵"""
        if self.scaler is None:
            raise ValueError("请先调用fit方法训练特征处理器")
        return self.scaler.transform(X)
    
    def transform_dict(self, features: Dict[str, float]) -> Dict[str, float]:
        """转换特征字典"""
        if self.scaler is None:
            raise ValueError("请先调用fit方法训练特征处理器")
            
        X = np.zeros((1, len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            if name in features:
                X[0, i] = features[name]
                
        X_scaled = self.scaler.transform(X)
        
        scaled_features = {}
        for i, name in enumerate(self.feature_names):
            if name in features:
                scaled_features[name] = X_scaled[0, i]
                
        return scaled_features
    
    def save(self, filename: str):
        """保存特征处理器"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filename)
        
    @classmethod
    def load(cls, filename: str) -> 'FeatureProcessor':
        """加载特征处理器"""
        data = joblib.load(filename)
        processor = cls()
        processor.scaler = data['scaler']
        processor.feature_names = data['feature_names']
        return processor 