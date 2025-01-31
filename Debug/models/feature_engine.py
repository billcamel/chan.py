"""统一的特征工程引擎"""
import math
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import talib
from enum import Enum, auto
import json

from Chan import CChan
from Common.CEnum import BI_DIR, BSP_TYPE, FX_TYPE, MACD_ALGO
from .feature_processor import FeatureProcessor

class FeatureType(Enum):
    """特征类型"""
    TECHNICAL = auto()  # 技术指标
    MARKET = auto()     # 市场特征
    PATTERN = auto()    # 形态特征
    CHAN = auto()    # 缠论特征
    CUSTOM = auto()     # 自定义特征

class FeatureEngine:
    """特征工程引擎"""
    def __init__(self):
        """初始化特征引擎
        
        Args:
            enabled_types: 启用的特征类型列表
        """
        self.enabled_types = [ FeatureType.TECHNICAL, FeatureType.CHAN]
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
        
        # # 检查数据长度
        # if len(df) < self.normalize_window:
        #     return pd.DataFrame()
            
        # 计算所有特征
        features = {}
        
        # 根据配置计算不同类型的特征
        if FeatureType.TECHNICAL in self.enabled_types:
            features.update(self._get_technical_features(df))
            
        if FeatureType.MARKET in self.enabled_types:
            market_features = self._get_market_features(df)
            features.update(market_features)
            
        if FeatureType.PATTERN in self.enabled_types:
            pattern_features = self._get_pattern_features(df)
            features.update(pattern_features)
        
        # 转换为单行DataFrame
        return pd.DataFrame([features])
    
    def get_features(self, kline_data: List[Any], idx: int, chan_snapshot: Any = None) -> Dict[str, float]:
        """获取某个时间点的所有特征
        
        Args:
            kline_data: K线数据列表
            idx: 当前K线索引
            chan_snapshot: 缠论快照对象，用于计算缠论相关特征
            
        Returns:
            特征字典
        """
        features = {}
        if idx < self.normalize_window:
            return features
            
        # 使用transform获取基础特征
        df = self.transform(kline_data)
        if not df.empty:
            # 获取基础特征（transform返回的是单行DataFrame）
            features = df.iloc[0].to_dict()
            features = {k: v for k, v in features.items() if pd.notna(v)}
        
        # 添加缠论特征
        if FeatureType.CHAN in self.enabled_types and chan_snapshot:
            features.update(self._get_chan_features(chan_snapshot))
            
        return features
    
    def _get_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算最后一根K线的技术指标特征"""
        features = {}
        
        try:
            # === 趋势类指标 ===
            # MA族
            ma_periods = [5, 10, 20, 60, 120]
            ma_values = {}  # 存储不同周期的均线值
            
            for period in ma_periods:
                sma = talib.SMA(df.close[-2*period:], timeperiod=period)
                ema = talib.EMA(df.close[-2*period:], timeperiod=period)
                wma = talib.WMA(df.close[-2*period:], timeperiod=period)
                
                ma_values[f'sma{period}'] = sma.iloc[-1]
                ma_values[f'ema{period}'] = ema.iloc[-1]
                
                # 基础均线值
                features[f'sma{period}'] = sma.iloc[-1]
                features[f'ema{period}'] = ema.iloc[-1]
                features[f'wma{period}'] = wma.iloc[-1]
                
                # 价格相对均线位置
                features[f'price_sma{period}_ratio'] = (df.close.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1]
                features[f'price_ema{period}_ratio'] = (df.close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
                
                # 均线斜率
                features[f'sma{period}_slope'] = (sma.iloc[-1] - sma.iloc[-2]) / sma.iloc[-2]
                features[f'ema{period}_slope'] = (ema.iloc[-1] - ema.iloc[-2]) / ema.iloc[-2]
            
            # 均线多空头排列
            for i in range(len(ma_periods)-1):
                period1 = ma_periods[i]
                period2 = ma_periods[i+1]
                # 短期均线相对长期均线位置
                features[f'sma{period1}_{period2}_ratio'] = (ma_values[f'sma{period1}'] - ma_values[f'sma{period2}']) / ma_values[f'sma{period2}']
                features[f'ema{period1}_{period2}_ratio'] = (ma_values[f'ema{period1}'] - ma_values[f'ema{period2}']) / ma_values[f'ema{period2}']
            
            # 均线多头/空头排列强度
            features['ma_bull_power'] = sum(1 for i in range(len(ma_periods)-1) 
                                          if ma_values[f'sma{ma_periods[i]}'] > ma_values[f'sma{ma_periods[i+1]}'])
            features['ma_bear_power'] = sum(1 for i in range(len(ma_periods)-1) 
                                          if ma_values[f'sma{ma_periods[i]}'] < ma_values[f'sma{ma_periods[i+1]}'])
            
            # MACD族
            macd, signal, hist = talib.MACD(df.close[-2*20:])
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_hist'] = hist.iloc[-1]
            
            # MACD相关指标
            features['macd_ratio'] = macd.iloc[-1] / df.close.iloc[-1]  # MACD相对价格
            features['macd_hist_ratio'] = hist.iloc[-1] / df.close.iloc[-1]  # MACD柱状图相对价格
            features['macd_cross'] = (macd.iloc[-1] - signal.iloc[-1]) / df.close.iloc[-1]  # MACD交叉状态
            features['macd_hist_slope'] = (hist.iloc[-1] - hist.iloc[-2]) / abs(hist.iloc[-2]) if hist.iloc[-2] != 0 else 0  # 柱状图斜率
            
            # 抛物线转向 - SAR
            sar = talib.SAR(df.high[-2*20:], df.low[-2*20:])
            features['sar'] = sar.iloc[-1]
            features['sar_ratio'] = (df.close.iloc[-1] - sar.iloc[-1]) / df.close.iloc[-1]
            features['sar_slope'] = (sar.iloc[-1] - sar.iloc[-2]) / sar.iloc[-2]
            
            # === 动量类指标 ===
            # RSI族
            for period in [12, 24, 60]:
                rsi = talib.RSI(df.close[-2*period:], timeperiod=period)
                features[f'rsi_{period}'] = rsi.iloc[-1]
                # RSI变化率
                features[f'rsi_{period}_slope'] = (rsi.iloc[-1] - rsi.iloc[-2]) / rsi.iloc[-2]
                # RSI超买超卖
                features[f'rsi_{period}_overbought'] = 1 if rsi.iloc[-1] > 70 else 0
                features[f'rsi_{period}_oversold'] = 1 if rsi.iloc[-1] < 30 else 0
            
            # 随机指标族
            slowk, slowd = talib.STOCH(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            fastk, fastd = talib.STOCHF(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            features['slowk'] = slowk.iloc[-1]
            features['slowd'] = slowd.iloc[-1]
            features['fastk'] = fastk.iloc[-1]
            features['fastd'] = fastd.iloc[-1]
            
            # KD指标交叉和超买超卖
            features['kd_cross'] = (slowk.iloc[-1] - slowd.iloc[-1])
            features['kd_overbought'] = 1 if slowk.iloc[-1] > 80 and slowd.iloc[-1] > 80 else 0
            features['kd_oversold'] = 1 if slowk.iloc[-1] < 20 and slowd.iloc[-1] < 20 else 0
            
            # 动量指标
            for period in [10, 20]:
                mom = talib.MOM(df.close[-2*period:], timeperiod=period)
                roc = talib.ROC(df.close[-2*period:], timeperiod=period)
                trix = talib.TRIX(df.close[-2*period:], timeperiod=period)
                
                features[f'mom_{period}'] = mom.iloc[-1]
                features[f'roc_{period}'] = roc.iloc[-1]
                features[f'trix_{period}'] = trix.iloc[-1]
            
            # === 波动类指标 ===
            # 布林带
            upper, middle, lower = talib.BBANDS(df.close[-2*20:])
            features['boll'] = middle.iloc[-1]
            features['boll_ub'] = upper.iloc[-1]
            features['boll_lb'] = lower.iloc[-1]
            features['boll_width'] = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            features['boll_position'] = (df.close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            # ATR族
            atr = talib.ATR(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            natr = talib.NATR(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            features['atr'] = atr.iloc[-1]
            features['natr'] = natr.iloc[-1]
            features['atr_ratio'] = atr.iloc[-1] / df.close.iloc[-1]
            
            # === 成交量类指标 ===
            # 成交量均线
            for period in [5, 10, 20]:
                vol_sma = talib.SMA(df.volume[-2*period:], timeperiod=period)
                features[f'volume_sma{period}'] = vol_sma.iloc[-1]
                features[f'volume_sma{period}_ratio'] = df.volume.iloc[-1] / vol_sma.iloc[-1]
            
            # 成交量动量指标
            obv = talib.OBV(df.close[-2*20:], df.volume[-2*20:])
            ad = talib.AD(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:], df.volume[-2*20:])
            adosc = talib.ADOSC(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:], df.volume[-2*20:])
            
            features['obv'] = obv.iloc[-1]
            features['ad'] = ad.iloc[-1]
            features['adosc'] = adosc.iloc[-1]
            
            # === 趋势确认指标 ===
            # DMI族
            plus_di = talib.PLUS_DI(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            minus_di = talib.MINUS_DI(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            adx = talib.ADX(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            adxr = talib.ADXR(df.high[-2*20:], df.low[-2*20:], df.close[-2*20:])
            
            features['plus_di'] = plus_di.iloc[-1]
            features['minus_di'] = minus_di.iloc[-1]
            features['adx'] = adx.iloc[-1]
            features['adxr'] = adxr.iloc[-1]
            
            # DMI趋势强度
            features['dmi_trend'] = (plus_di.iloc[-1] - minus_di.iloc[-1]) * adx.iloc[-1] / 100
            features['dmi_cross'] = (plus_di.iloc[-1] - minus_di.iloc[-1])
            features['dmi_trend_strength'] = 1 if adx.iloc[-1] > 25 else 0
            
        except Exception as e:
            print(f"计算技术指标出错: {str(e)}")
            
        return features
    
    def _get_market_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算最后一根K线的市场特征"""
        features = {}
        
        try:
            # # 价格归一化
            # close_norm = self._normalize_series(df.close)
            # high_norm = self._normalize_series(df.high)
            # low_norm = self._normalize_series(df.low)
            
            # features['norm_close'] = close_norm.iloc[-1]
            # features['norm_high'] = high_norm.iloc[-1]
            # features['norm_low'] = low_norm.iloc[-1]
            
            # # 波动率特征
            # for period in [5, 10, 20]:
            #     volatility = df.close.rolling(period).std() / df.close
            #     features[f'volatility_{period}'] = volatility.iloc[-1]
                
            # # 趋势特征
            # for period in [5, 10, 20]:
            #     # 价格趋势
            #     price_trend = (df.close.iloc[-1] - df.close.iloc[-period]) / df.close.iloc[-period]
            #     features[f'price_trend_{period}'] = price_trend
                
            #     # 成交量趋势
            #     volume_trend = (df.volume.iloc[-1] - df.volume.iloc[-period]) / df.volume.iloc[-period]
            #     features[f'volume_trend_{period}'] = volume_trend
            
            # # 价格区间特征
            # for period in [5, 10, 20]:
            #     period_high = df.high.rolling(period).max().iloc[-1]
            #     period_low = df.low.rolling(period).min().iloc[-1]
            #     cur_price = df.close.iloc[-1]
                
            #     # 当前价格在区间的位置
            #     price_position = (cur_price - period_low) / (period_high - period_low) if period_high != period_low else 0.5
            #     features[f'price_position_{period}'] = price_position
                
            #     # 区间宽度
            #     range_width = (period_high - period_low) / cur_price
            #     features[f'range_width_{period}'] = range_width
            
            # 成交量特征
            volume_ma = df.volume[-2*20:].rolling(20).mean()
            features['volume_ratio'] = df.volume.iloc[-1] / volume_ma.iloc[-1]
            features['volume_ma_slope'] = (volume_ma.iloc[-1] - volume_ma.iloc[-2]) / volume_ma.iloc[-2]
            
            # 振幅特征
            features['amplitude'] = (df.high.iloc[-1] - df.low.iloc[-1]) / df.open.iloc[-1]
            features['amplitude_ma5'] = df[-2*5:].apply(
                lambda x: (x['high'] - x['low']) / x['open'], 
                axis=1
            ).rolling(5).mean().iloc[-1]
            
            # 涨跌幅特征
            features['return'] = (df.close[-2*20:].iloc[-1] - df.open[-2*20:].iloc[-1]) / df.open[-2*20:].iloc[-1]
            features['return_ma5'] = ((df.close[-2*20:] - df.open[-2*20:]) / df.open[-2*20:]).rolling(5).mean().iloc[-1]
            
        except Exception as e:
            print(f"计算市场特征出错: {str(e)}")
            
        return features
    
    def _get_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算最后一根K线的形态特征"""
        features = {}
        
        try:
            # 蜡烛图形态
            pattern_funcs = [
                talib.CDLDOJI,           # 十字星
                talib.CDLENGULFING,      # 吞没形态
                talib.CDLHARAMI,         # 孕线
                talib.CDLHAMMER,         # 锤子线
                talib.CDLMARUBOZU,       # 光头光脚/长实体
                talib.CDLHANGINGMAN,     # 上吊线
                talib.CDLMORNINGSTAR,    # 晨星
                talib.CDLEVENINGSTAR,    # 暮星
            ]
            
            for func in pattern_funcs:
                pattern_name = func.__name__.replace('CDL', '').lower()
                pattern_value = func(df.open, df.high, df.low, df.close)
                features[f'pattern_{pattern_name}'] = pattern_value.iloc[-1]
            
            # K线形态特征
            features['body_size'] = abs(df.close.iloc[-1] - df.open.iloc[-1]) / df.open.iloc[-1]
            features['upper_shadow'] = (df.high.iloc[-1] - max(df.open.iloc[-1], df.close.iloc[-1])) / df.open.iloc[-1]
            features['lower_shadow'] = (min(df.open.iloc[-1], df.close.iloc[-1]) - df.low.iloc[-1]) / df.open.iloc[-1]
            
            # 连续K线形态
            features['three_white_soldiers'] = 1 if (
                df.close.iloc[-3:] > df.open.iloc[-3:]).all() else 0
            features['three_black_crows'] = 1 if (
                df.close.iloc[-3:] < df.open.iloc[-3:]).all() else 0
                
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
        stop=False
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
            # 记录废弃的特征数
            discarded_features = 0
            for feature_name, value in features:
                if isinstance(value, bool):
                    value = 1 if value else 0
                elif not isinstance(value, (int, float)):
                    discarded_features += 1
                    continue
                if feature_name in feature_meta:
                    feat_idx = feature_meta[feature_name]
                    X[i, feat_idx] = float(value)
            if discarded_features > 0:
                print(f"废弃了 {discarded_features} 个非数值特征")
            
            # 记录标记点
            plot_marker[feature_info["open_time"].to_str()] = (
                "√" if y[i] else "×", 
                "down" if feature_info["is_buy"] else "up"
            )
        
        # 打印一些调试信息
        # print(f"\n特征维度: {X.shape}")
        # print(f"特征名列表: {list(feature_meta.keys())}")
        # print(f"样本数量: {n_samples}")
        # print(f"正样本数量: {np.sum(y)}")
        # print(f"正样本比例: {np.mean(y):.2%}")
        
        # 检查特征矩阵是否有效
        if np.any(np.isnan(X)):
            print("警告: 特征矩阵包含NaN值")
        if np.any(np.isinf(X)):
            print("警告: 特征矩阵包含Inf值")
            
        return plot_marker, feature_meta, X, y

    def _get_chan_features(self, chan_snapshot: CChan) -> Dict[str, float]:
        """计算缠论特征
        
        Args:
            chan_snapshot: 缠论快照对象
            
        Returns:
            缠论特征字典
        """
        features = {}
        
        try:
            # 获取买卖点列表
            bsp_list = chan_snapshot.get_bsp(0)
            if not bsp_list or len(bsp_list) < 4:
                print("买卖点太少，不计算缠论特征：", len(bsp_list))
                features.update({
                    'bsp_d1': 0,
                    'bsp_d2': 0,
                    'bsp_d3': 0,
                    'bsp_count': 0,
                    'last_bsp_type': 0
                })
                features['fx_type'] = 0
                features['fx_high'] = 0
                features['fx_low'] = 0
                
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
                # features['bi_macd_amount'] = 0
                features['bi_macd_volumn_avg'] = 0
                # features['bi_macd_amount_avg'] = 0
                # features['bi_macd_turnrate_avg'] = 0
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
                # print("没有分型")
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
                # features['bi_macd_amount'] = last_bi.cal_macd_metric(MACD_ALGO.AMOUNT, False)
                features['bi_macd_volumn_avg'] = last_bi.cal_macd_metric(MACD_ALGO.VOLUMN_AVG, False)
                # features['bi_macd_amount_avg'] = last_bi.cal_macd_metric(MACD_ALGO.AMOUNT_AVG, False)
                # features['bi_macd_turnrate_avg'] = last_bi.cal_macd_metric(MACD_ALGO.TURNRATE_AVG, False)
            else:
                # print("没有笔")
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
                # features['bi_macd_amount'] = 0
                features['bi_macd_volumn_avg'] = 0
                # features['bi_macd_amount_avg'] = 0
                # features['bi_macd_turnrate_avg'] = 0
                
        except Exception as e:
            print(f"计算缠论特征出错: {str(e)}")
            
        return features

# 创建全局特征引擎实例
feature_engine = FeatureEngine() 