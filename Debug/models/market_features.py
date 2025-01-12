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
    processor = FeatureProcessor()
    processor.fit(X, list(feature_meta.keys()))
    X = processor.transform(X)
    
    # 保存特征处理器和特征meta
    processor.save("feature_processor.joblib")
    with open("feature.meta", "w") as fid:
        json.dump(feature_meta, fid, indent=2)
        
    return plot_marker, feature_meta, X, y 