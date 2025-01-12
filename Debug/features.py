"""特征工程相关的公共函数"""
import numpy as np
import talib
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import json

def safe_div(a, b, default=0):
    """安全除法,避免除0错误"""
    try:
        if abs(b) < 1e-10:  # 分母太小
            return default
        return float(a) / float(b)
    except:
        return default

def get_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    try:
        # 趋势指标
        for period in [5, 10, 20, 60]:
            df[f'sma{period}'] = talib.SMA(df.close, timeperiod=period)
            df[f'ema{period}'] = talib.EMA(df.close, timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df.close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # RSI
        for period in [6, 12, 24]:
            df[f'rsi_{period}'] = talib.RSI(df.close, timeperiod=period)
        
        # 布林带
        upper, middle, lower = talib.BBANDS(df.close)
        df['boll'] = middle
        df['boll_ub'] = upper
        df['boll_lb'] = lower
        df['boll_width'] = (upper - lower) / middle
        df['boll_position'] = (df.close - lower) / (upper - lower)
        
        # KDJ
        slowk, slowd = talib.STOCH(df.high, df.low, df.close)
        df['kdj_k'] = slowk
        df['kdj_d'] = slowd
        df['kdj_j'] = 3 * slowk - 2 * slowd
        
        # 成交量
        df['volume_sma5'] = talib.SMA(df.volume, timeperiod=5)
        df['volume_sma10'] = talib.SMA(df.volume, timeperiod=10)
        df['obv'] = talib.OBV(df.close, df.volume)
        df['ad'] = talib.AD(df.high, df.low, df.close, df.volume)
        df['volume_delta'] = df.volume.diff()
        df['volume_relative'] = df.volume / df.volume.rolling(window=20).mean()
        
        # ATR
        df['atr'] = talib.ATR(df.high, df.low, df.close)
        df['atr_ratio'] = df['atr'] / df.close
        
        # DMI
        df['plus_di'] = talib.PLUS_DI(df.high, df.low, df.close)
        df['minus_di'] = talib.MINUS_DI(df.high, df.low, df.close)
        df['adx'] = talib.ADX(df.high, df.low, df.close)
        
        # 动量指标
        df['cci'] = talib.CCI(df.high, df.low, df.close)
        df['mfi'] = talib.MFI(df.high, df.low, df.close, df.volume)
        df['roc'] = talib.ROC(df.close)
        df['willr'] = talib.WILLR(df.high, df.low, df.close)
        
        return df
        
    except Exception as e:
        print(f"计算技术指标出错: {str(e)}")
        return df

class StockFeatureEngine:
    """股票特征工程引擎"""
    
    def __init__(self):
        self.feature_processor = get_technical_features
        
    def transform(self, kline_data: List) -> pd.DataFrame:
        """转换K线数据为特征DataFrame"""
        # 转换为DataFrame
        df = pd.DataFrame({
            'open': [kl.open for kl in kline_data],
            'high': [kl.high for kl in kline_data],
            'low': [kl.low for kl in kline_data],
            'close': [kl.close for kl in kline_data],
            'volume': [kl.trade_info.metric['volume'] for kl in kline_data]
        })
        
        # 计算技术指标
        try:
            df = self.feature_processor(df)
        except Exception as e:
            print(f"特征处理出错: {str(e)}")
            
        return df
        
    def get_features(self, kline_data: List, idx: int) -> Dict[str, float]:
        """获取某个时间点的所有特征"""
        df = self.transform(kline_data)
        features = df.iloc[idx].to_dict()
        # 移除NaN值
        return {k: v for k, v in features.items() if pd.notna(v)}

class FeatureProcessor:
    """特征处理器"""
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """训练特征处理器
        
        Args:
            X: 特征矩阵
            feature_names: 特征名列表
        """
        self.feature_names = feature_names
        # 使用RobustScaler对异常值不敏感
        self.scaler = RobustScaler()
        self.scaler.fit(X)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换特征矩阵
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 归一化后的特征矩阵
        """
        if self.scaler is None:
            raise ValueError("请先调用fit方法训练特征处理器")
        return self.scaler.transform(X)
    
    def transform_dict(self, features: Dict[str, float]) -> Dict[str, float]:
        """转换特征字典
        
        Args:
            features: 特征字典
            
        Returns:
            Dict[str, float]: 归一化后的特征字典
        """
        if self.scaler is None:
            raise ValueError("请先调用fit方法训练特征处理器")
            
        # 构建特征向量
        X = np.zeros((1, len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            if name in features:
                X[0, i] = features[name]
                
        # 归一化
        X_scaled = self.scaler.transform(X)
        
        # 转回字典
        scaled_features = {}
        for i, name in enumerate(self.feature_names):
            if name in features:
                scaled_features[name] = X_scaled[0, i]
                
        return scaled_features
    
    def save(self, filename: str):
        """保存特征处理器"""
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filename)
        
    @classmethod
    def load(cls, filename: str) -> 'FeatureProcessor':
        """加载特征处理器"""
        import joblib
        data = joblib.load(filename)
        processor = cls()
        processor.scaler = data['scaler']
        processor.feature_names = data['feature_names']
        return processor

# 创建全局特征引擎实例
feature_engine = StockFeatureEngine()

def get_market_features(kline_data: List[Any], idx: int) -> Dict[str, float]:
    """获取市场特征
    
    Args:
        kline_data: K线数据列表
        idx: 当前K线索引
        
    Returns:
        dict: 特征字典
    """
    cur_kl = kline_data[idx]
    prev_kl = kline_data[idx-1] if idx > 0 else cur_kl
    
    # 基础价格特征
    features = {
        # K线基本形态特征
        "price_change": safe_div(cur_kl.close - prev_kl.close, prev_kl.close),
        "amplitude": safe_div(cur_kl.high - cur_kl.low, cur_kl.open),
        "upper_shadow": safe_div(cur_kl.high - max(cur_kl.open, cur_kl.close), cur_kl.open),
        "lower_shadow": safe_div(min(cur_kl.open, cur_kl.close) - cur_kl.low, cur_kl.open),
        "body_size": safe_div(abs(cur_kl.close - cur_kl.open), cur_kl.open),
    }
    
    # 获取技术指标特征
    if idx >= 33:  # 确保有足够的历史数据
        ta_features = feature_engine.get_features(kline_data, idx)
        features.update(ta_features)
    
    return features

def save_features(bsp_dict: Dict, bsp_academy: List[int]):
    """保存特征到numpy格式
    
    Args:
        bsp_dict: 买卖点特征字典
        bsp_academy: 标准买卖点列表
        
    Returns:
        plot_marker: 标记点信息
        feature_meta: 特征元信息
        X: 特征矩阵
        y: 标签数组
    """
    # 收集所有特征名
    feature_meta = {}
    cur_feature_idx = 0
    for _, feature_info in bsp_dict.items():
        for feature_name, _ in feature_info['feature'].items():
            if feature_name not in feature_meta:
                feature_meta[feature_name] = cur_feature_idx
                cur_feature_idx += 1
    
    # 准备特征矩阵和标签
    n_samples = len(bsp_dict)
    n_features = len(feature_meta)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    plot_marker = {}
    
    # 填充特征矩阵
    for i, (bsp_klu_idx, feature_info) in enumerate(bsp_dict.items()):
        # 生成label
        y[i] = float(bsp_klu_idx in bsp_academy)
        
        # 填充特征
        features = feature_info['feature'].items()
        for feature_name, value in features:
            if not isinstance(value, (int, float)):
                continue
            if feature_name in feature_meta:
                feat_idx = feature_meta[feature_name]
                X[i, feat_idx] = float(value)
        
        # 记录标记点
        plot_marker[feature_info["open_time"].to_str()] = (
            "√" if y[i] else "×", 
            "down" if feature_info["is_buy"] else "up"
        )
    
    # 打印一些调试信息
    print(f"\n特征维度: {X.shape}")
    print(f"特征名列表: {list(feature_meta.keys())}")
    print(f"样本数量: {n_samples}")
    print(f"正样本数量: {np.sum(y)}")
    print(f"正样本比例: {np.mean(y):.2%}")
    
    # 检查特征矩阵是否有效
    if np.any(np.isnan(X)):
        print("警告: 特征矩阵包含NaN值")
    if np.any(np.isinf(X)):
        print("警告: 特征矩阵包含Inf值")
    
    # 特征归一化
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # 保存特征meta
    with open("feature.meta", "w") as fid:
        json.dump(feature_meta, fid, indent=2)
        
    return plot_marker, feature_meta, X, y 