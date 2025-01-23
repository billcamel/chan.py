"""统一的特征工程引擎"""
import math
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import talib
from enum import Enum, auto
import json

from Common.CEnum import BI_DIR, BSP_TYPE, FX_TYPE, MACD_ALGO
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
        self.enabled_types = [FeatureType.TECHNICAL, FeatureType.MARKET, FeatureType.PATTERN]
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
        
        # 检查数据长度
        if len(df) < self.normalize_window:
            return pd.DataFrame()
            
        features_df = pd.DataFrame()
        
        # 根据配置计算不同类型的特征
        if FeatureType.TECHNICAL in self.enabled_types:
            features_df = pd.concat([features_df, self._get_technical_features(df)], axis=1)
            
        if FeatureType.MARKET in self.enabled_types:
            features_df = pd.concat([features_df, self._get_market_features(df)], axis=1)
            
        if FeatureType.PATTERN in self.enabled_types:
            features_df = pd.concat([features_df, self._get_pattern_features(df)], axis=1)
            
        return features_df
    
    def get_features(self, kline_data: List[Any], idx: int, chan_snapshot: Any = None) -> Dict[str, float]:
        """获取某个时间点的所有特征
        
        Args:
            kline_data: K线数据列表
            idx: 当前K线索引
            chan_snapshot: 缠论快照对象，用于计算缠论相关特征
            
        Returns:
            特征字典
        """
        if idx < self.normalize_window:
            return {}
            
        # 使用transform获取基础特征
        df = self.transform(kline_data)
        if df.empty:
            return {}
            
        # 获取基础特征
        features = df.iloc[idx].to_dict()
        features = {k: v for k, v in features.items() if pd.notna(v)}
        
        # 添加缠论特征
        if FeatureType.PATTERN in self.enabled_types and chan_snapshot:
            features.update(self._get_chan_features(chan_snapshot))
            
        return features
    
    def _get_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        features = pd.DataFrame()
        
        try:
            # === 趋势类指标 ===
            # MA族
            for period in [60, 120]:
                features[f'sma{period}'] = talib.SMA(df.close, timeperiod=period)
                features[f'ema{period}'] = talib.EMA(df.close, timeperiod=period)
                features[f'wma{period}'] = talib.WMA(df.close, timeperiod=period)  # 加权移动平均
            
            # MACD族
            macd, signal, hist = talib.MACD(df.close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # 抛物线转向 - SAR
            features['sar'] = talib.SAR(df.high, df.low)
            features['sar_ratio'] = (df.close - features['sar']) / df.close
            
            # === 动量类指标 ===
            # RSI族
            for period in [ 12, 24, 60]:
                features[f'rsi_{period}'] = talib.RSI(df.close, timeperiod=period)
            
            # 随机指标族
            slowk, slowd = talib.STOCH(df.high, df.low, df.close)
            fastk, fastd = talib.STOCHF(df.high, df.low, df.close)
            features['slowk'] = slowk
            features['slowd'] = slowd
            features['fastk'] = fastk
            features['fastd'] = fastd
            
            # 动量指标
            for period in [10, 20]:
                features[f'mom_{period}'] = talib.MOM(df.close, timeperiod=period)
                features[f'roc_{period}'] = talib.ROC(df.close, timeperiod=period)
                features[f'trix_{period}'] = talib.TRIX(df.close, timeperiod=period)
            
            # === 波动类指标 ===
            # 布林带
            upper, middle, lower = talib.BBANDS(df.close)
            features['boll'] = middle
            features['boll_ub'] = upper
            features['boll_lb'] = lower
            features['boll_width'] = (upper - lower) / middle
            features['boll_position'] = (df.close - lower) / (upper - lower)
            
            # ATR族
            features['atr'] = talib.ATR(df.high, df.low, df.close)
            features['natr'] = talib.NATR(df.high, df.low, df.close)  # 归一化ATR
            features['atr_ratio'] = features['atr'] / df.close
            
            # === 成交量类指标 ===
            # 成交量均线
            for period in [5, 10, 20]:
                features[f'volume_sma{period}'] = talib.SMA(df.volume, timeperiod=period)
            
            # 成交量动量指标
            features['obv'] = talib.OBV(df.close, df.volume)  # 能量潮
            features['ad'] = talib.AD(df.high, df.low, df.close, df.volume)  # 累积/派发线
            features['adosc'] = talib.ADOSC(df.high, df.low, df.close, df.volume)  # A/D震荡指标
            
            # === 趋势确认指标 ===
            # DMI族
            features['plus_di'] = talib.PLUS_DI(df.high, df.low, df.close)
            features['minus_di'] = talib.MINUS_DI(df.high, df.low, df.close)
            features['adx'] = talib.ADX(df.high, df.low, df.close)
            features['adxr'] = talib.ADXR(df.high, df.low, df.close)
            
            # === 其他综合指标 ===
            # 威廉指标
            features['willr'] = talib.WILLR(df.high, df.low, df.close)
            
            # CCI
            features['cci'] = talib.CCI(df.high, df.low, df.close)
            
            # 资金流向指标
            features['mfi'] = talib.MFI(df.high, df.low, df.close, df.volume)
            
            # 相对强弱指数
            features['dx'] = talib.DX(df.high, df.low, df.close)  # 动向指数
            
            # 价格动量指标
            features['ppo'] = talib.PPO(df.close)  # 价格震荡百分比
            features['ultosc'] = talib.ULTOSC(df.high, df.low, df.close)  # 终极波动指标
            
            # === 自定义组合指标 ===
            # 均线交叉
            features['ma_cross'] = (features['sma10'] - features['sma20']) / df.close
            features['di_cross'] = features['plus_di'] - features['minus_di']
            
            # 趋势强度
            features['trend_strength'] = features['adx'] * np.sign(features['di_cross'])
            
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

    def _get_chan_features(self, chan_snapshot: Any) -> Dict[str, float]:
        """计算缠论特征
        
        Args:
            chan_snapshot: 缠论快照对象
            
        Returns:
            缠论特征字典
        """
        features = {}
        
        try:
            # 获取买卖点列表
            bsp_list = chan_snapshot.get_bsp()
            if not bsp_list or len(bsp_list) < 4:
                features.update({
                    'bsp_d1': 0,
                    'bsp_d2': 0,
                    'bsp_d3': 0,
                    'bsp_count': 0,
                    'last_bsp_type': 0
                })
                return features
                
            # 获取最后一个买卖点
            last_bsp = bsp_list[-1]
            
            # 计算最近4个买卖点之间的距离特征
            last_bsp2 = bsp_list[-2]
            last_bsp3 = bsp_list[-3] 
            last_bsp4 = bsp_list[-4]
            
            # 计算相邻买卖点之间的欧氏距离
            dk1 = last_bsp.klu.idx - last_bsp2.klu.idx # k线距离
            dp1 = abs(last_bsp.klu.close - last_bsp2.klu.close) # 价格差
            features['bsp_d1'] = math.sqrt(dp1*dp1 + dk1*dk1)
            
            dk2 = last_bsp2.klu.idx - last_bsp3.klu.idx
            dp2 = abs(last_bsp2.klu.close - last_bsp3.klu.close)
            features['bsp_d2'] = math.sqrt(dp2*dp2 + dk2*dk2)
            
            dk3 = last_bsp3.klu.idx - last_bsp4.klu.idx
            dp3 = abs(last_bsp3.klu.close - last_bsp4.klu.close)
            features['bsp_d3'] = math.sqrt(dp3*dp3 + dk3*dk3)
            
            # 添加买卖点统计特征
            features['bsp_count'] = len(bsp_list)
            features['last_bsp_type'] = 1 if last_bsp.is_buy else -1
            
            # 获取当前级别的缠论对象
            cur_lv_chan = chan_snapshot[0]
            
            # 添加分型特征
            if cur_lv_chan[-2].fx:
                features['fx_type'] = 1 if cur_lv_chan[-2].fx == FX_TYPE.BOTTOM else -1
                features['fx_high'] = cur_lv_chan[-2].high
                features['fx_low'] = cur_lv_chan[-2].low
            else:
                features['fx_type'] = 0
                features['fx_high'] = 0
                features['fx_low'] = 0
                
            # 添加笔的特征
            if cur_lv_chan.bi_list:
                last_bi = cur_lv_chan.bi_list[-1]
                features['bi_direction'] = 1 if last_bi.dir == BI_DIR.UP else -1
                features['bi_length'] = last_bi.get_klu_cnt()
                features['bi_amp'] = last_bi.amp()  
                features['bi_is_sure'] = last_bi.is_sure
                features['bi_macd_area'] = last_bi.cal_macd_metric(MACD_ALGO.AREA, False)
                features['bi_macd_diff'] = last_bi.cal_macd_metric(MACD_ALGO.DIFF, False)
                features['bi_macd_slope'] = last_bi.cal_macd_metric(MACD_ALGO.SLOPE, False)
                features['bi_macd_amp'] = last_bi.cal_macd_metric(MACD_ALGO.AMP, False)
                features['bi_macd_peak'] = last_bi.cal_macd_metric(MACD_ALGO.PEAK, False)
                features['bi_macd_full_area'] = last_bi.cal_macd_metric(MACD_ALGO.FULL_AREA, False)
                features['bi_macd_volumn'] = last_bi.cal_macd_metric(MACD_ALGO.VOLUMN, False)
                features['bi_macd_amount'] = last_bi.cal_macd_metric(MACD_ALGO.AMOUNT, False)
                features['bi_macd_volumn_avg'] = last_bi.cal_macd_metric(MACD_ALGO.VOLUMN_AVG, False)
                features['bi_macd_amount_avg'] = last_bi.cal_macd_metric(MACD_ALGO.AMOUNT_AVG, False)
                features['bi_macd_turnrate_avg'] = last_bi.cal_macd_metric(MACD_ALGO.TURNRATE_AVG, False)
            else:
                features['bi_direction'] = 0
                features['bi_length'] = 0
                features['bi_amp'] = 0
                features['bi_is_sure'] = 0
                features['bi_macd_area'] = 0
                features['bi_macd_diff'] = 0
                features['bi_macd_slope'] = 0
                features['bi_macd_amp'] = 0
                features['bi_macd_peak'] = 0
                features['bi_macd_full_area'] = 0
                features['bi_macd_volumn'] = 0
                features['bi_macd_amount'] = 0
                features['bi_macd_volumn_avg'] = 0
                features['bi_macd_amount_avg'] = 0
                features['bi_macd_turnrate_avg'] = 0
                
        except Exception as e:
            print(f"计算缠论特征出错: {str(e)}")
            
        return features

# 创建全局特征引擎实例
feature_engine = FeatureEngine() 