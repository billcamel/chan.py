"""市场基础特征"""
from typing import Dict, List, Any
import numpy as np
import json
from .stock_features import feature_engine
from .feature_processor import FeatureProcessor

def safe_div(a, b, default=0):
    """安全除法"""
    try:
        if abs(b) < 1e-10:
            return default
        return float(a) / float(b)
    except:
        return default

def normalize_price(price_list: List[float], window: int = 20) -> List[float]:
    """对价格序列进行归一化处理
    
    Args:
        price_list: 价格序列
        window: 归一化窗口大小
        
    Returns:
        归一化后的价格序列
    """
    normalized = []
    for i in range(len(price_list)):
        start_idx = max(0, i - window + 1)
        window_prices = price_list[start_idx:i + 1]
        if not window_prices:
            normalized.append(0)
            continue
        window_mean = np.mean(window_prices)
        window_std = np.std(window_prices)
        if window_std == 0:
            normalized.append(0)
        else:
            normalized.append((price_list[i] - window_mean) / window_std)
    return normalized

def get_market_features(kline_data: List[Any], idx: int) -> Dict[str, float]:
    """获取市场特征
    
    Args:
        kline_data: K线数据列表
        idx: 当前K线索引
        
    Returns:
        dict: 特征字典
    """
    if idx < 20:  # 需要至少20根K线的历史数据
        return {}
        
    # 提取价格序列
    close_prices = [k.close for k in kline_data[:idx+1]]
    high_prices = [k.high for k in kline_data[:idx+1]]
    low_prices = [k.low for k in kline_data[:idx+1]]
    
    # 归一化处理
    norm_close = normalize_price(close_prices)
    norm_high = normalize_price(high_prices)
    norm_low = normalize_price(low_prices)
    
    # 计算技术指标时使用归一化后的价格
    features = {
        'norm_close': norm_close[-1],
        'norm_high': norm_high[-1],
        'norm_low': norm_low[-1],
        'price_std_20': np.std(norm_close[-20:]),
        'price_trend_20': (norm_close[-1] - norm_close[-20]) if len(norm_close) >= 20 else 0,
        # 'vol_std_20': np.std([k.trade_info.metric['volume'] for k in kline_data[idx-19:idx+1]]) if idx >= 19 else 0,
        # 'vol_trend_20': (kline_data[idx].trade_info.metric['volume'] - kline_data[idx-19].trade_info.metric['volume']) if idx >= 19 else 0,
    }
    
    # 添加移动平均相关特征
    for period in [5, 10, 20]:
        if len(norm_close) >= period:
            ma = np.mean(norm_close[-period:])
            features.update({
                f'ma{period}': ma,
                f'ma{period}_slope': (ma - np.mean(norm_close[-period-1:-1])) if len(norm_close) > period else 0,
                f'price_ma{period}_diff': norm_close[-1] - ma
            })
    
    # 添加波动率特征
    for period in [5, 10, 20]:
        if len(norm_high) >= period and len(norm_low) >= period:
            features[f'atr{period}'] = np.mean([
                norm_high[i] - norm_low[i] 
                for i in range(-period, 0)
            ])
    
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
    processor = FeatureProcessor()
    processor.fit(X, list(feature_meta.keys()))
    X = processor.transform(X)
    
    # 保存特征处理器和特征meta
    processor.save("feature_processor.joblib")
    with open("feature.meta", "w") as fid:
        json.dump(feature_meta, fid, indent=2)
        
    return plot_marker, feature_meta, X, y 