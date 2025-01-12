"""股票技术指标特征"""
import pandas as pd
import talib
from typing import Dict, List, Any

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
        df = pd.DataFrame({
            'open': [kl.open for kl in kline_data],
            'high': [kl.high for kl in kline_data],
            'low': [kl.low for kl in kline_data],
            'close': [kl.close for kl in kline_data],
            'volume': [kl.trade_info.metric['volume'] for kl in kline_data]
        })
        
        try:
            df = self.feature_processor(df)
        except Exception as e:
            print(f"特征处理出错: {str(e)}")
            
        return df
        
    def get_features(self, kline_data: List, idx: int) -> Dict[str, float]:
        """获取某个时间点的所有特征"""
        df = self.transform(kline_data)
        features = df.iloc[idx].to_dict()
        return {k: v for k, v in features.items() if pd.notna(v)}

# 创建全局特征引擎实例
feature_engine = StockFeatureEngine() 