"""统一的特征工程引擎"""
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import talib
from enum import Enum, auto
import json
from .feature_processor import FeatureProcessor

class FeatureType(Enum):
    """特征类型"""
    TECHNICAL = auto()  # 技术指标
    MARKET = auto()     # 市场特征
    PATTERN = auto()    # 形态特征
    CUSTOM = auto()     # 自定义特征

class FeatureEngine:
    """特征工程引擎"""
    def __init__(self):
        """初始化特征引擎
        
        Args:
            enabled_types: 启用的特征类型列表
        """
        self.enabled_types = [FeatureType.TECHNICAL, FeatureType.MARKET]
        # 固定参数
        self.normalize_window = 100  # 归一化窗口
        
    def transform(self, kline_data: List[Any]) -> pd.DataFrame:
        """转换K线数据为特征DataFrame
        
        Args:
            kline_data: K线数据列表
            
        Returns:
            特征DataFrame
        """
        # 基础数据准备
        df = pd.DataFrame({
            'open': [kl.open for kl in kline_data],
            'high': [kl.high for kl in kline_data],
            'low': [kl.low for kl in kline_data],
            'close': [kl.close for kl in kline_data],
            'volume': [kl.trade_info.metric.get('volume', 0) for kl in kline_data]
        })
        
        features_df = pd.DataFrame()
        
        # 根据配置计算不同类型的特征
        if FeatureType.TECHNICAL in self.enabled_types:
            features_df = pd.concat([features_df, self._get_technical_features(df)], axis=1)
            
        if FeatureType.MARKET in self.enabled_types:
            features_df = pd.concat([features_df, self._get_market_features(df)], axis=1)
            
        if FeatureType.PATTERN in self.enabled_types:
            features_df = pd.concat([features_df, self._get_pattern_features(df)], axis=1)
            
        return features_df
    
    def get_features(self, kline_data: List[Any], idx: int) -> Dict[str, float]:
        """获取某个时间点的所有特征
        
        Args:
            kline_data: K线数据列表
            idx: 当前K线索引
            
        Returns:
            特征字典
        """
        if idx < self.normalize_window:
            return {}
            
        df = self.transform(kline_data)
        features = df.iloc[idx].to_dict()
        return {k: v for k, v in features.items() if pd.notna(v)}
    
    def _get_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        features = pd.DataFrame()
        
        try:
            # 趋势指标
            for period in [10, 20, 60]:
                features[f'sma{period}'] = talib.SMA(df.close, timeperiod=period)
                features[f'ema{period}'] = talib.EMA(df.close, timeperiod=period)
            
            # MACD
            macd, signal, hist = talib.MACD(df.close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # RSI
            for period in [12]:
                features[f'rsi_{period}'] = talib.RSI(df.close, timeperiod=period)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(df.close)
            features['boll'] = middle
            features['boll_ub'] = upper
            features['boll_lb'] = lower
            features['boll_width'] = (upper - lower) / middle
            features['boll_position'] = (df.close - lower) / (upper - lower)
            
            # KDJ
            slowk, slowd = talib.STOCH(df.high, df.low, df.close)
            features['kdj_k'] = slowk
            features['kdj_d'] = slowd
            features['kdj_j'] = 3 * slowk - 2 * slowd
            
            # 成交量
            features['volume_sma5'] = talib.SMA(df.volume, timeperiod=5)
            features['volume_sma10'] = talib.SMA(df.volume, timeperiod=10)
            features['obv'] = talib.OBV(df.close, df.volume)
            features['ad'] = talib.AD(df.high, df.low, df.close, df.volume)
            
            # ATR
            features['atr'] = talib.ATR(df.high, df.low, df.close)
            features['atr_ratio'] = features['atr'] / df.close
            
            # DMI
            features['plus_di'] = talib.PLUS_DI(df.high, df.low, df.close)
            features['minus_di'] = talib.MINUS_DI(df.high, df.low, df.close)
            features['adx'] = talib.ADX(df.high, df.low, df.close)
            
            # 动量指标
            features['cci'] = talib.CCI(df.high, df.low, df.close)
            features['mfi'] = talib.MFI(df.high, df.low, df.close, df.volume)
            features['roc'] = talib.ROC(df.close)
            features['willr'] = talib.WILLR(df.high, df.low, df.close)
            
        except Exception as e:
            print(f"计算技术指标出错: {str(e)}")
            
        return features
    
    def _get_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市场特征"""
        features = pd.DataFrame()
        
        try:
            # 价格归一化
            close_norm = self._normalize_series(df.close)
            high_norm = self._normalize_series(df.high)
            low_norm = self._normalize_series(df.low)
            
            features['norm_close'] = close_norm
            features['norm_high'] = high_norm
            features['norm_low'] = low_norm
            
            # 波动率特征
            for period in [ 20]:
                features[f'volatility_{period}'] = df.close.rolling(period).std() / df.close
                
            # 趋势特征
            # for period in [5, 10, 20]:
            #     features[f'trend_{period}'] = (df.close - df.close.shift(period)) / df.close.shift(period)
                
            # 成交量特征
            features['volume_ratio'] = df.volume / df.volume.rolling(20).mean()
            features['volume_trend'] = (df.volume - df.volume.shift(1)) / df.volume.shift(1)
            
        except Exception as e:
            print(f"计算市场特征出错: {str(e)}")
            
        return features
    
    def _get_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算形态特征"""
        features = pd.DataFrame()
        
        try:
            # 蜡烛图形态
            pattern_funcs = [
                talib.CDLDOJI,
                talib.CDLENGULFING,
                talib.CDLHARAMI,
                talib.CDLHAMMER,
                talib.CDLSHOOTINGSTAR
            ]
            
            for func in pattern_funcs:
                pattern_name = func.__name__.replace('CDL', '').lower()
                features[f'pattern_{pattern_name}'] = func(
                    df.open, df.high, df.low, df.close
                )
                
        except Exception as e:
            print(f"计算形态特征出错: {str(e)}")
            
        return features
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """归一化序列数据"""
        return (series - series.rolling(self.normalize_window).mean()) / \
               series.rolling(self.normalize_window).std()
    
    def save_features(self, bsp_dict: Dict, bsp_academy: List[int]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
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
        processor = FeatureProcessor()
        processor.fit(X, list(feature_meta.keys()))
        X = processor.transform(X)
        
        # 保存特征处理器和特征meta
        processor.save("feature_processor.joblib")
        with open("feature.meta", "w") as fid:
            json.dump(feature_meta, fid, indent=2)
            
        return plot_marker, feature_meta, X, y

# 创建全局特征引擎实例
feature_engine = FeatureEngine() 