"""特征工程相关的公共函数"""
import numpy as np
import talib as ta
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, RobustScaler

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

def safe_div(a, b, default=0):
    """安全除法,避免除0错误"""
    try:
        if abs(b) < 1e-10:  # 分母太小
            return default
        return float(a) / float(b)
    except:
        return default

def get_ta_features(kline_data: List[Any], idx: int) -> Dict[str, float]:
    """使用TA-Lib计算技术指标特征
    
    Args:
        kline_data: K线数据列表
        idx: 当前K线索引
        
    Returns:
        dict: 技术指标特征字典
    """
    if idx < 33:  # 需要足够的历史数据来计算指标
        return {}
        
    # 准备数据
    close = np.array([k.close for k in kline_data[max(0, idx-100):idx+1]])
    high = np.array([k.high for k in kline_data[max(0, idx-100):idx+1]])
    low = np.array([k.low for k in kline_data[max(0, idx-100):idx+1]])
    volume = np.array([k.trade_info.metric['volume'] for k in kline_data[max(0, idx-100):idx+1]])
    
    features = {}
    
    try:
        # 趋势指标
        sma_periods = [5, 10, 20, 60]
        for period in sma_periods:
            sma = ta.SMA(close, timeperiod=period)
            features[f'sma{period}'] = sma[-1]
            features[f'sma{period}_slope'] = (sma[-1] - sma[-2]) / sma[-2] if len(sma) > 1 else 0
        
        # MACD指标
        macd, macd_signal, macd_hist = ta.MACD(close)
        features.update({
            'macd': macd[-1],
            'macd_signal': macd_signal[-1],
            'macd_hist': macd_hist[-1],
            'macd_hist_slope': (macd_hist[-1] - macd_hist[-2]) / abs(macd_hist[-2]) if len(macd_hist) > 1 and macd_hist[-2] != 0 else 0
        })
        
        # RSI指标
        for period in [6, 12, 24]:
            rsi = ta.RSI(close, timeperiod=period)
            features[f'rsi{period}'] = rsi[-1]
        
        # 布林带指标
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=20)
        features.update({
            'bb_upper': upperband[-1],
            'bb_middle': middleband[-1],
            'bb_lower': lowerband[-1],
            'bb_width': (upperband[-1] - lowerband[-1]) / middleband[-1],
            'bb_position': (close[-1] - lowerband[-1]) / (upperband[-1] - lowerband[-1]) if upperband[-1] != lowerband[-1] else 0.5
        })
        
        # KDJ指标
        slowk, slowd = ta.STOCH(high, low, close)
        features.update({
            'kdj_k': slowk[-1],
            'kdj_d': slowd[-1],
            'kdj_j': 3 * slowk[-1] - 2 * slowd[-1]
        })
        
        # 成交量指标
        features.update({
            'volume_sma5': ta.SMA(volume, timeperiod=5)[-1],
            'volume_sma10': ta.SMA(volume, timeperiod=10)[-1],
            'obv': ta.OBV(close, volume)[-1],
            'ad': ta.AD(high, low, close, volume)[-1]
        })
        
        # ATR指标
        atr = ta.ATR(high, low, close, timeperiod=14)
        features['atr'] = atr[-1]
        features['atr_ratio'] = atr[-1] / close[-1]
        
        # DMI指标
        plus_di = ta.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = ta.MINUS_DI(high, low, close, timeperiod=14)
        adx = ta.ADX(high, low, close, timeperiod=14)
        features.update({
            'plus_di': plus_di[-1],
            'minus_di': minus_di[-1],
            'adx': adx[-1]
        })
        
    except Exception as e:
        print(f"计算技术指标出错: {str(e)}")
        return {}
        
    return features

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
    
    # 趋势特征
    lookback = min(20, idx+1)  # 最多看前20根K线
    if lookback >= 2:
        hist_kls = kline_data[idx-lookback+1:idx+1]
        high_prices = [kl.high for kl in hist_kls]
        low_prices = [kl.low for kl in hist_kls]
        close_prices = [kl.close for kl in hist_kls]
        
        features.update({
            "price_highest_ratio": safe_div(cur_kl.close - max(high_prices[:-1]), max(high_prices[:-1])),
            "price_lowest_ratio": safe_div(cur_kl.close - min(low_prices[:-1]), min(low_prices[:-1])),
            "trend_strength": safe_div(close_prices[-1] - close_prices[0], close_prices[0]),
            "volatility": safe_div(max(high_prices) - min(low_prices), min(low_prices))
        })
    
    # 添加TA-Lib计算的技术指标特征
    ta_features = get_ta_features(kline_data, idx)
    features.update(ta_features)
    
    # 确保所有特征都是float类型
    for k, v in features.items():
        try:
            features[k] = float(v)
        except (TypeError, ValueError):
            features[k] = 0.0
            
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
    import json
    
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
    
    # 保存特征处理器
    processor.save("feature_processor.joblib")
    
    # 保存特征meta
    with open("feature.meta", "w") as fid:
        json.dump(feature_meta, fid, indent=2)
        
    return plot_marker, feature_meta, X, y 