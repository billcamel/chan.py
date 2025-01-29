import numpy as np
import functools
from Common.CEnum import FX_TYPE, TREND_TYPE


class CFeatureGenerator:
    global feature_registry, feature_registry_ext
    feature_registry = {"pattern": [], "kline": [], "indicator": [], "chan": []}
    feature_registry_ext = {"last": [], "pre": [], "bsp": [], "other": [], "chan": []}

    def register_factor(category):
        def decorator(func):
            print(f"注册因子：{func.__name__}，类别：{category}")
            feature_registry[category].append(func)
            return func
        return decorator

    def register_feature(category):
        def decorator(func):
            print(f"注册特征：{func.__name__}，类别：{category}")
            feature_registry_ext[category].append(func)
            return func
        return decorator

    def __init__(self):
        self.feature_funcs = []
        self.last_klu = None
        self.pre_klu = None
        self.last_bsp = None
        self.pre_bsp = None
        self.last_seg_bsp = None
        self.pre_seg_bsp = None
        self.last_bsp_klu = None
        self.cur_lv_chan = None
        self.last_klc = None
        self.seg_bsp_lst = None
        self.bi_lst = None
        self.re_klu_lst = None
        self.klu_lst = None

    #================================
    # 数据初始化
    #================================
    def _initialize_data(self, chan_snapshot):
        """依次调用各初始化方法来设置通用数据"""
        self._set_cur_lv_chan(chan_snapshot)
        self._set_last_klc()
        self._set_klu_lst(80)
        self._set_last_klu()
        self._set_pre_klu()
        self._set_last_bsp_klu()
        self._set_bsp_lst(chan_snapshot)
        self._set_seg_bsp_lst(chan_snapshot)
        self._set_bi_lst()

    def _set_cur_lv_chan(self, chan_snapshot):
        """设置当前通道数据"""
        self.cur_lv_chan = chan_snapshot[0] if chan_snapshot else None
    def _set_last_klc(self):
        """设置最后一个KLC数据"""
        self.last_klc = self.cur_lv_chan[-1] if self.cur_lv_chan else None
    def _set_klu_lst(self, length=20):
        """设置KLU列表"""
        re_klu_lst = []
        last_klu = self.last_klu
        while len(re_klu_lst) < length and last_klu:
            re_klu_lst.append(last_klu)
            last_klu = last_klu.pre
        self.re_klu_lst = re_klu_lst
        self.klu_lst = re_klu_lst[::-1]
        self.re_klu_lst = tuple(re_klu_lst)
        self.klu_lst = tuple(self.klu_lst)

    def _set_last_klu(self):
        """设置最后一个KLU数据"""
        self.last_klu = self.last_klc[-1] if self.last_klc else None
    def _set_pre_klu(self):
        """设置倒数第二个KLU数据"""
        self.pre_klu = self.last_klu.pre if self.last_klu else None
    def _set_last_bsp_klu(self):
        """设置最后一个买卖点的KLU"""
        self.last_bsp_klu = self.last_bsp.klu if self.last_bsp else None
    def _set_bsp_lst(self, chan_snapshot):
        """设置买卖点列表和最后一个买卖点"""
        self.bsp_lst = chan_snapshot.get_bsp() if chan_snapshot else []
        self.last_bsp = self.bsp_lst[-1] if self.bsp_lst else None
        self.pre_bsp = self.bsp_lst[-2] if len(self.bsp_lst) > 1 else None
    def _set_seg_bsp_lst(self, chan_snapshot):
        """设置分段买卖点列表和最后一个分段买卖点"""
        # self.seg_bsp_lst = chan_snapshot.get_seg_bsp() if chan_snapshot else []
        self.seg_bsp_lst = []
        self.last_seg_bsp = self.seg_bsp_lst[-1] if self.seg_bsp_lst else None
        self.pre_seg_bsp = self.seg_bsp_lst[-2] if len(self.seg_bsp_lst) > 1 else None
    def _set_bi_lst(self):
        """设置笔列表和前一笔数据"""
        self.bi_lst = self.cur_lv_chan.bi_list if self.cur_lv_chan else []

    def ma_dic(self, klu):
        """获取MA指标字典"""
        ma_dic = klu.trend[TREND_TYPE.MEAN]
        return ma_dic
    
    def max_dic(self, klu):
        """获取MAX指标字典"""
        max_dic = klu.trend[TREND_TYPE.MAX]
        return max_dic
    
    def min_dic(self, klu):
        """获取MIN指标字典"""
        min_dic = klu.trend[TREND_TYPE.MIN]
        return min_dic

    #================================
    # 特征处理方法
    #================================
    def add_feature(self, feature_func):
        """动态添加特征函数，feature_func 是返回特征的函数。"""
        self.feature_funcs.append(feature_func)

    def generate_features(self, chan_snapshot):
        """为每个买卖点生成特征"""
        self._initialize_data(chan_snapshot)  # 初始化数据
        feature_dict = {}  # 改用字典存储所有特征
        for idx, feature_func in enumerate(self.feature_funcs):
            # 打印原始特征函数名称
            original_func_name = feature_func.__closure__[0].cell_contents.__name__ if feature_func.__closure__ else "Unknown"
            result = feature_func()
            if result:
                feature_dict.update(result)  # 将每个特征函数的结果合并到同一个字典中
        return feature_dict  # 返回合并后的特征字典

    def add_all_features(self, skip=None):
        """遍历所有类别的注册特征，添加特征函数到 feature_funcs 列表中并打印信息"""
        for category, features in feature_registry_ext.items():
            if skip and category == skip:
                    continue

            # 将特征函数添加到 feature_funcs 列表中
            for feature_func in features:
                print(f"添加特征: {feature_func.__name__}，类别: {category}")
                # 这里使用一个函数来绑定当前的 feature_func
                self.add_feature(self._bind_feature(feature_func))
        print(f"添加特征数: {len(self.feature_funcs)}")

    def _bind_feature(self, feature_func):
        """为特征函数绑定 self 参数，确保每个函数独立运行"""
        return lambda: feature_func(self)

    
    #================================
    # 指标因子
    #================================
    @register_factor("kline")
    def body_rate(self, klu):
        """计算实体与高低比率"""
        close_open = klu.close - klu.open
        high_low = klu.high - klu.low
        body_rate = close_open / high_low if high_low != 0 else None
        return body_rate

    @register_factor("kline")
    def wick_rate(self, klu):
        """计算上下影线比率"""
        if klu.close > klu.open:
            up_wick = klu.high - klu.close + 1e-7
            down_wick = klu.open - klu.low + 1e-7
        else:
            up_wick = klu.high - klu.open + 1e-7
            down_wick = klu.close - klu.low + 1e-7
        wick_rate = down_wick / up_wick
        return wick_rate 

    @register_factor("kline")

    def range_rate(self, range1, range2):
        """计算两K线的波动率比"""
        range_rate = range1 / range2 if range2 != 0 else None
        return range_rate

    @register_factor("kline")

    def range(self, high, low):
        """计算两K线的波动率比"""
        range = abs(high - low) / low
        return range

    @register_factor("pattern")

    def ma1_color(self, pre_klu, last_klu, length):
        """计算MA1颜色变化"""
        if self.pre_klu is None or self.last_klu is None:
            return None
        pre_ma_dic = self.ma_dic(pre_klu)
        last_ma_dic = self.ma_dic(last_klu)
        pre_ma = pre_ma_dic[length]
        last_ma = last_ma_dic[length]
        ma1_color = (last_ma - pre_ma) > 0
        return ma1_color

    @register_factor("pattern")

    def ma2_color(self, pre_klu, last_klu, length):
        """计算MA2颜色变化"""
        if self.pre_klu is None or self.last_klu is None:
            return None
        pre_ma_dic = self.ma_dic(pre_klu)
        last_ma_dic = self.ma_dic(last_klu)
        pre_ma = pre_ma_dic[length]
        last_ma = last_ma_dic[length]
        ma2_color = (last_ma - pre_ma) > 0
        return ma2_color

    @register_factor("pattern")

    def ma3_color(self, pre_klu, last_klu, length):
        """计算MA3颜色变化"""
        if self.pre_klu is None or self.last_klu is None:
            return None
        pre_ma_dic = self.ma_dic(pre_klu)
        last_ma_dic = self.ma_dic(last_klu)
        pre_ma = pre_ma_dic[length]
        last_ma = last_ma_dic[length]
        ma3_color = (last_ma - pre_ma) > 0
        return ma3_color
    
    @register_factor("indicator")

    def rsi(self, klu):
        """计算RSI指标"""
        return klu.rsi
    
    @register_factor("indicator")

    def neutral_rsi(self, klu):
        """计算中性RSI指标"""
        rsi = klu.rsi
        neutral_rsi = abs(rsi - 0.5)*2 / 100
        return neutral_rsi

    @register_factor("indicator")

    def cci(self, klu, length):
        """计算CCI指标"""
        ma_dic = self.ma_dic(klu)
        ma = ma_dic[length]
        hlc3 = (klu.high + klu.low + klu.close) / 3
        mean_dev = abs(hlc3 - ma)
        cci = (hlc3 - ma) / (0.015 * mean_dev) if mean_dev != 0 else 0
        return cci

    @register_factor("indicator")

    def boll(self, klu, length):
        """计算Bollinger带"""
        boll_up = klu.bolls[length].UP
        boll_down = klu.bolls[length].DOWN
        boll_mid = klu.bolls[length].MID
        return (boll_up, boll_mid, boll_down)

    @register_factor("indicator")

    def bbb(self, klu, length):
        """计算Bollinger带宽比"""
        boll_up = klu.bolls[length].UP
        boll_down = klu.bolls[length].DOWN
        return (klu.close - boll_down) / (boll_up - boll_down)

    @register_factor("indicator")

    def normalize_bbw(self, klu, length):
        """计算标准化的BBW"""
        boll_up = klu.bolls[length].UP
        boll_down = klu.bolls[length].DOWN
        boll_mid = klu.bolls[length].MID
        bbw = (boll_up - boll_down) / boll_mid * 100
        normalize_bbw = bbw / 100
        return normalize_bbw

    @register_factor("indicator")

    def normalize_bbt(self, klu, length1, length2):
        """计算标准化的BBT"""
        boll1_up = klu.bolls[length1].UP
        boll1_down = klu.bolls[length1].DOWN
        boll1_mid = klu.bolls[length1].MID
        boll2_up = klu.bolls[length2].UP
        boll2_down = klu.bolls[length2].DOWN
        # boll2_mid = klu.bolls[length2].MID
        bbt = (abs(boll1_down - boll2_down) -
               abs(boll1_up - boll2_up)) / boll1_mid * 100
        normalize_bbt = bbt / 100
        return normalize_bbt

    @register_factor("indicator")

    def ma_ratio(self, klu):
        """计算MA比率"""
        if klu is None or klu.boll is None:
            return None
        try:
            ma_ratio = (klu.boll.MID - klu.bolls[50].MID) / klu.bolls[50].MID
        except AttributeError:
            return None
        return ma_ratio

    @register_factor("indicator")

    def roc(self, re_klu_lst, n):
        """计算给定KLU列表的ROC（Rate of Change）特征，返回所有ROC值的列表"""
        roc_lst = []
        # 如果数据量小于n，直接返回空列表
        if len(re_klu_lst) < n:
            return None
        for i in range(len(re_klu_lst) - n):
            last_klu = re_klu_lst[i]
            # 获取n期前的收盘价
            close_n_periods_ago = re_klu_lst[i + n].close
            
            # 计算当前收盘价与n期前收盘价的差值
            roc = ((last_klu.close - close_n_periods_ago) / close_n_periods_ago) * 100 if close_n_periods_ago != 0 else 0
            roc_lst.append(roc)
        
        return roc_lst

    @register_factor("indicator")

    def adx(self, re_klu_lst, adx_length=14, di_length=14):
        """
        计算 ADX 指标，并返回每个周期的 ADX 值序列。
        """
        if len(re_klu_lst) < di_length:
            # print(f"数据长度 {len(re_klu_lst)} 不足以计算 ADX，返回 NaN")
            return None
        klu_lst = re_klu_lst[::-1]  # 将 K-line 列表反转
        # 初始化列表来存储 DM 和 TR 的平滑值
        smoothed_plus_dm = []
        smoothed_minus_dm = []
        smoothed_tr = []
        adx_lst = []
        for i in range(1, len(klu_lst)):  # 从第二根K线开始计算
            current_klu = klu_lst[i]
            prev_klu = klu_lst[i - 1]
            # 计算 +DM 和 -DM
            up_move = current_klu.high - prev_klu.high
            down_move = prev_klu.low - current_klu.low
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0
            # 计算 TR
            true_range = max(current_klu.high - current_klu.low,
                            abs(current_klu.high - prev_klu.close),
                            abs(current_klu.low - prev_klu.close))
            # 平滑 DM 和 TR
            if len(smoothed_plus_dm) < di_length:
                smoothed_plus_dm.append(plus_dm)
                smoothed_minus_dm.append(minus_dm)
                smoothed_tr.append(true_range)
            else:
                smoothed_plus_dm.append(smoothed_plus_dm[-1] * (di_length - 1) / di_length + plus_dm / di_length)
                smoothed_minus_dm.append(smoothed_minus_dm[-1] * (di_length - 1) / di_length + minus_dm / di_length)
                smoothed_tr.append(smoothed_tr[-1] * (di_length - 1) / di_length + true_range / di_length)
        # 计算 +DI 和 -DI
        plus_di = [100 * smoothed_plus_dm[i] / smoothed_tr[i] if smoothed_tr[i] != 0 else 0 for i in range(di_length, len(smoothed_tr))]
        minus_di = [100 * smoothed_minus_dm[i] / smoothed_tr[i] if smoothed_tr[i] != 0 else 0 for i in range(di_length, len(smoothed_tr))]
        # 计算 DX
        dx_lst = []
        for p_di, m_di in zip(plus_di, minus_di):
            dx = 100 * abs(p_di - m_di) / (p_di + m_di) if (p_di + m_di) != 0 else 0
            dx_lst.append(dx)
        # 计算 ADX 列表
        if len(dx_lst) < adx_length:
            # print(f"DX 数据 {len(dx_lst)} 条，不足以计算 ADX，返回 NaN")
            return None
        # 计算初始的 ADX
        adx = np.mean(dx_lst[:adx_length])  # 初始ADX
        adx_lst.append(adx)
        # 逐步平滑ADX并生成序列
        for dx in dx_lst[adx_length:]:
            adx = ((adx * (adx_length - 1)) + dx) / adx_length
            adx_lst.append(adx)
        adx_lst = adx_lst[::-1]  # 反转 ADX 列表
        return adx_lst

    #================================
    # 指标特征
    #================================
    @register_feature("last")
    def ma1_color_feature(self):
        return {"ma1_color": self.ma1_color(self.pre_klu, self.last_klu, 5)}
    @register_feature("last")
    def ma2_color_feature(self):
        return {"ma2_color": self.ma1_color(self.pre_klu, self.last_klu, 20)}
    @register_feature("last")
    def ma3_color_feature(self):
        return {"ma3_color": self.ma1_color(self.pre_klu, self.last_klu, 80)}
    
    @register_feature("last")
    def last_roc_feature(self):
        roc_lst = self.roc(self.re_klu_lst, 9)
        if not roc_lst:
            return None
        return {"last_roc": roc_lst[0]}
    @register_feature("pre")
    def pre_roc_feature(self):
        roc_lst = self.roc(self.re_klu_lst, 9)
        if not roc_lst or len(roc_lst) < 2:
            return None
        return {"pre_roc": roc_lst[1]}
    
    @register_feature("last")
    def last_normalize_adx_feature(self):
        adx_lst = self.adx(self.re_klu_lst, 14, 14)
        if not adx_lst:
            return None
        return {"last_adx": adx_lst[0] / 100}
    @register_feature("pre")
    def pre_normalize_adx_feature(self):
        adx_lst = self.adx(self.re_klu_lst, 14, 14)
        if not adx_lst or len(adx_lst) < 2:
            return None
        return {"pre_adx": adx_lst[1] / 100}
    
    @register_feature("last")
    def last_body_rate_feature(self):
        if not self.last_klu:
            return None
        return {"last_body_rate": self.body_rate(self.last_klu)}
    @register_feature("pre")
    def pre_body_rate_feature(self):
        if not self.pre_klu:
            return None
        return {"pre_body_rate": self.body_rate(self.pre_klu)}
    @register_feature("bsp")
    def last_bsp_body_rate_feature(self):
        if not self.last_bsp_klu:
            return None
        return {"last_bsp_body_rate": self.body_rate(self.last_bsp_klu)}

    @register_feature("last")
    def last_wick_rate_feature(self):
        if not self.last_klu:
            return None
        return {"last_wick_rate": self.wick_rate(self.last_klu)}
    @register_feature("pre")
    def pre_wick_rate_feature(self):
        if not self.pre_klu:
            return None
        return {"pre_wick_rate": self.wick_rate(self.pre_klu)}
    @register_feature("bsp")
    def last_bsp_wick_rate_feature(self):
        if not self.last_bsp_klu:
            return None
        return {"last_bsp_wick_rate": self.wick_rate(self.last_bsp_klu)}

    @register_feature("last")
    def last_klu_range_rate_feature(self):
        if not self.last_klu:
            return None
        last_klu_range = self.range(self.last_klu.high, self.last_klu.low)
        pre_klu_range = self.range(self.pre_klu.high, self.pre_klu.low)
        last_klu_range_rate = self.range_rate(pre_klu_range, last_klu_range)
        return {"last_klu_range_rate": last_klu_range_rate}

    @register_feature("chan")
    def klc23_range_rate_feature(self):
        if len(self.cur_lv_chan) < 3:
            return None
        klc3 = self.cur_lv_chan[-1]
        klc2 = self.cur_lv_chan[-2]
        klc1 = self.cur_lv_chan[-3]
        klc1_range = self.range(klc1.high, klc1.low)
        klc2_range = self.range(klc2.high, klc2.low)
        klc3_range = self.range(klc3.high, klc3.low)
        klc23_range_rate = self.range_rate(klc2_range, klc3_range)
        return {"klc23_range_rate": klc23_range_rate}

    @register_feature("chan")
    def klc21_range_rate_feature(self):
        if len(self.cur_lv_chan) < 3:
            return None
        klc3 = self.cur_lv_chan[-1]
        klc2 = self.cur_lv_chan[-2]
        klc1 = self.cur_lv_chan[-3]
        klc1_range = self.range(klc1.high, klc1.low)
        klc2_range = self.range(klc2.high, klc2.low)
        klc21_range_rate = self.range_rate(klc2_range, klc1_range)
        return {"klc21_range_rate": klc21_range_rate}

    @register_feature("chan")
    def klc13_range_rate_feature(self):
        if len(self.cur_lv_chan) < 3:
            return None
        klc3 = self.cur_lv_chan[-1]
        klc2 = self.cur_lv_chan[-2]
        klc1 = self.cur_lv_chan[-3]
        klc1_range = self.range(klc1.high, klc1.low)
        klc2_range = self.range(klc2.high, klc2.low)
        klc3_range = self.range(klc3.high, klc3.low)
        klc13_range_rate = self.range_rate(klc1_range, klc3_range)
        return {"klc13_range_rate": klc13_range_rate}

    @register_feature("chan")
    def fx_factor_feature(self):
        if not self.cur_lv_chan or (len(self.cur_lv_chan) < 3):
            return None
        fx_factor = self.fx_factor(self.cur_lv_chan)
        return {"fx_factor": fx_factor}

    @register_feature("last")
    def last_normalize_kdj_feature(self):
        if not self.last_klu:
            return None
        kdj = self.last_klu.kdj
        return {"last_normalize_kdj": kdj.k / 100}
    @register_feature("pre")
    def pre_normalize_kdj_feature(self):
        if not self.pre_klu:
            return None
        kdj = self.pre_klu.kdj
        return {"pre_normalize_kdj": kdj.k / 100}
    @register_feature("bsp")
    def last_bsp_normalize_kdj_feature(self):
        if not self.last_bsp_klu:
            return None
        kdj = self.last_bsp_klu.kdj
        return {"last_bsp_normalize_kdj": kdj.k / 100}

    @register_feature("last")
    def last_normalize_rsi_feature(self):
        if not self.last_klu:
            return None
        rsi = self.last_klu.rsi
        return {"last_normalize_rsi": rsi / 100}
    @register_feature("pre")
    def pre_normalize_rsi_feature(self):
        if not self.pre_klu:
            return None
        rsi = self.pre_klu.rsi
        return {"pre_normalize_rsi": rsi / 100}
    @register_feature("bsp")
    def last_bsp_normalize_rsi_feature(self):
        if not self.last_bsp_klu:
            return None
        rsi = self.last_bsp_klu.rsi
        return {"last_bsp_normalize_rsi": rsi / 100}

    @register_feature("last")
    def last_neutral_rsi_feature(self):
        neutral_rsi = self.neutral_rsi(self.last_klu)
        return {"last_neutral_rsi": neutral_rsi}
    @register_feature("pre")
    def pre_neutral_rsi_feature(self):
        neutral_rsi = self.neutral_rsi(self.pre_klu)
        return {"pre_neutral_rsi": neutral_rsi}
    @register_feature("bsp")
    def last_bsp_neutral_rsi_feature(self):
        if not self.last_bsp_klu:
            return None
        neutral_rsi = self.neutral_rsi(self.last_bsp_klu)
        return {"last_bsp_neutral_rsi": neutral_rsi}

    @register_feature("last")
    def last_normalize_cci_feature(self):
        cci = self.cci(self.last_klu, 20)
        return {"last_normalize_cci": cci / 100}
    @register_feature("pre")
    def pre_normalize_cci_feature(self):
        cci = self.cci(self.pre_klu, 20)
        return {"pre_normalize_cci": cci / 100}
    @register_feature("bsp")
    def last_bsp_normalize_cci_feature(self):
        if not self.last_bsp_klu:
            return None
        cci = self.cci(self.last_bsp_klu, 20)
        return {"last_bsp_normalize_cci": cci / 100}
    
    @register_feature("last")
    def last_bbb_feature(self):
        bbb = self.bbb(self.last_klu, 20)
        return {"last_bbb": bbb}
    @register_feature("pre")
    def pre_bbb_feature(self):
        bbb = self.bbb(self.pre_klu, 20)
        return {"last_pre_bbb": bbb}
    @register_feature("bsp")
    def last_bsp_bbb_feature(self):
        if not self.last_bsp_klu:
            return None
        bbb = self.bbb(self.last_bsp_klu, 20)
        return {"last_bsp_bbb": bbb}
    
    @register_feature("last")
    def last_normalize_bbw_feature(self):
        return {"last_normalize_bbw": self.normalize_bbw(self.last_klu, 20)}
    @register_feature("pre")
    def pre_normalize_bbw_feature(self):
        return {"pre_normalize_bbw": self.normalize_bbw(self.pre_klu, 20)}
    @register_feature("bsp")
    def last_bsp_normalize_bbw_feature(self):
        if not self.last_bsp_klu:
            return None
        return {"last_bsp_normalize_bbw": self.normalize_bbw(self.last_bsp_klu, 20)}
    
    @register_feature("last")
    def last_normalize_bbt_feature(self):
        return {"last_normalize_bbt": self.normalize_bbt(self.last_klu, 20, 50)}
    @register_feature("pre")
    def pre_normalize_bbt_feature(self):
        return {"pre_normalize_bbt": self.normalize_bbt(self.pre_klu, 20, 50)}
    @register_feature("bsp")
    def last_bsp_normalize_bbt_feature(self):
        if not self.last_bsp_klu:
            return None
        return {"last_bsp_normalize_bbt": self.normalize_bbt(self.last_bsp_klu, 20, 50)}

    @register_feature("last")
    def last_ma_ratio_feature(self):
        return {"last_ma_ratio": self.ma_ratio(self.last_klu)}
    @register_feature("pre")
    def pre_ma_ratio_feature(self):
        return {"pre_ma_ratio": self.ma_ratio(self.pre_klu)}
    @register_feature("bsp")
    def last_bsp_ma_ratio_feature(self):
        if not self.last_bsp_klu:
            return None
        return {"last_bsp_ma_ratio": self.ma_ratio(self.last_bsp_klu)}

    #================================
    # 缠论因子
    #================================
    @register_factor("chan")

    def fx_factor(self, cur_lv_chan):
        """打印倒数三根 KLC 数据并计算 fx_factor"""
        # 打印函数被调用的信息
        # print("fx_factor 函数被调用")
        # print(f"  当前 cur_lv_chan 长度: {len(cur_lv_chan)}")
        # 检查 cur_lv_chan 长度是否足够
        if len(cur_lv_chan) < 3:
            # print("  数据不足以计算 fx_factor: cur_lv_chan 长度小于 3")
            return None
        # 获取倒数三根 KLC 数据
        klc3 = cur_lv_chan[-1]
        klc2 = cur_lv_chan[-2]
        klc1 = cur_lv_chan[-3]
        # print(f"  倒数第三根 KLC 数据: {klc1}")
        # print(f"  倒数第二根 KLC 数据: {klc2}")
        # print(f"  倒数第一根 KLC 数据: {klc3}")
        # 初始化 fx_factor
        fx_factor = None
        # 检查 klc3 的 fx 类型
        if klc2.fx == FX_TYPE.UNKNOWN:
            # print("  fx 类型未知 (FX_TYPE.UNKNOWN)，无法计算 fx_factor")
            return None
        elif klc2.fx == FX_TYPE.BOTTOM:
            fx_factor = (klc3.high - klc1.high) / klc1.high
            # print(f"  fx 类型为 BOTTOM，klc3.high: {klc3.high}, klc1.high: {klc1.high}, 计算 fx_factor: {fx_factor}")
        elif klc2.fx == FX_TYPE.TOP:
            fx_factor = (klc1.low - klc3.low) / klc1.low
            # print(f"  fx 类型为 TOP，klc1.low: {klc1.low}, klc3.low: {klc3.low}, 计算 fx_factor: {fx_factor}")
        # else:
        #     print(f"  未处理的 fx 类型: {klc3.fx}")
        return fx_factor

    @register_factor("chan")

    def chan_bsp_type(self, bsp):
        bsp_type = bsp.type[0].value
        bsp_type_mapping = {
            "1": 1, "1p": 1.5, "2": 2, "2s": 2.5,
            "3a": 3, "3b": 3.5
        }
        bsp_type = bsp_type_mapping.get(bsp_type, 0)  # 根据类型映射特征值
        return bsp_type

    @register_factor("chan")

    def chan_is_BiAndSeg_bsp(self, bsp, seg_bsp):
        """最后一个分段买卖点的类型特征"""
        seg_bsp_time = seg_bsp.klu.time
        bsp_time = bsp.klu.time
        is_BiAndSeg_bsp = True if bsp_time == seg_bsp_time else False
        return is_BiAndSeg_bsp

    # @register_factor("chan")

    # def bi_volume_rate(self, bi1, bi2):
    #     """计算两笔成交量比率"""
    #     if not (bi1 and bi2):
    #         return None
    #     bi1_total_volume = bi1.total_volume
    #     bi2_total_volume = bi2.total_volume
    #     bi_volume_rate = (bi1_total_volume / bi2_total_volume
    #             if bi2_total_volume != 0 else None)
    #     return bi_volume_rate

    @register_factor("chan")

    def chan_bsp_isB1(self, bsp):
        """当前买卖点的类型特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1 = bsp_type == "1"
        is_buy = bsp.is_buy
        bsp_isB1 = True if bsp_is1 and is_buy else False
        return bsp_isB1

    @register_factor("chan")

    def chan_bsp_isB1p(self, bsp):
        """当前买卖点的类型特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1p = bsp_type == "1p"
        is_buy = bsp.is_buy
        bsp_isB1p = True if bsp_is1p and is_buy else False
        return bsp_isB1p

    @register_factor("chan")

    def chan_bsp_isS1(self, bsp):
        """当前买卖点的类型特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1 = bsp_type == "1"
        is_buy = bsp.is_buy
        bsp_isS1 = True if bsp_is1 and not is_buy else False
        return bsp_isS1

    @register_factor("chan")

    def chan_bsp_isS1p(self, bsp):
        """当前买卖点的类型特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1p = bsp_type == "1p"
        is_buy = bsp.is_buy
        bsp_isS1p = True if bsp_is1p and not is_buy else False
        return  bsp_isS1p

    @register_factor("chan")

    def chan_bsp_isB2(self, bsp):
        """当前买卖点的B2特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2 = bsp_type == "2"
        is_buy = bsp.is_buy
        bsp_isB2 = True if bsp_is2 and is_buy else False
        return bsp_isB2

    @register_factor("chan")

    def chan_bsp_isS2(self, bsp):
        """当前买卖点的S2特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2 = bsp_type == "2"
        is_buy = bsp.is_buy
        bsp_isS2 = True if bsp_is2 and not is_buy else False
        return bsp_isS2

    @register_factor("chan")

    def chan_bsp_isB2s(self, bsp):
        """当前买卖点的B2s特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2s = bsp_type == "2s"
        is_buy = bsp.is_buy
        bsp_isB2s = True if bsp_is2s and is_buy else False
        return bsp_isB2s

    @register_factor("chan")

    def chan_bsp_isS2s(self, bsp):
        """当前买卖点的S2s特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2s = bsp_type == "2s"
        is_buy = bsp.is_buy
        bsp_isS2s = True if bsp_is2s and not is_buy else False
        return bsp_isS2s

    @register_factor("chan")

    def chan_bsp_isB3a(self, bsp):
        """判断买卖点是否为类型B3a的特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3a = bsp_type == "3a"
        is_buy = bsp.is_buy
        bsp_isB3a = True if bsp_is3a and is_buy else False
        return bsp_isB3a

    @register_factor("chan")

    def chan_bsp_isS3a(self, bsp):
        """判断买卖点是否为类型S3a的特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3a = bsp_type == "3a"
        is_buy = bsp.is_buy
        bsp_isS3a = True if bsp_is3a and not is_buy else False
        return bsp_isS3a

    @register_factor("chan")

    def chan_bsp_isB3b(self, bsp):
        """判断买卖点是否为类型B3b的特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3b = bsp_type == "3b"
        is_buy = bsp.is_buy
        bsp_isB3b = True if bsp_is3b and is_buy else False
        return bsp_isB3b

    @register_factor("chan")

    def chan_bsp_isS3b(self, bsp):
        """判断买卖点是否为类型S3b的特征"""
        bsp_type = bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3b = bsp_type == "3b"
        is_buy = bsp.is_buy
        bsp_isS3b = True if bsp_is3b and not is_buy else False
        return bsp_isS3b

    #================================
    # 缠论特征
    #================================
    # @register_feature("chan")
    # def lastBimid_volume_rate_feature(self):
    #     """计算当前笔和下一个笔的成交量比率"""
    #     if (not self.bi_lst) or (len(self.bi_lst) < 2):
    #         return None
    #     last_bi = self.bi_lst[-1]
    #     mid_bi = last_bi.pre
    #     pre_bi = mid_bi.pre

    #     lastBimid_volume_rate = self.bi_volume_rate(last_bi, mid_bi)
    #     return {"lastBimid_volume_rate": lastBimid_volume_rate}

    # @register_feature("chan")
    # def lastBipre_volume_rate_feature(self):
    #     """计算当前笔和下一个笔的成交量比率"""
    #     if (not self.bi_lst) or (len(self.bi_lst) < 3):
    #         return None
    #     last_bi = self.bi_lst[-1]
    #     mid_bi = last_bi.pre
    #     pre_bi = mid_bi.pre

    #     lastBipre_volume_rate = self.bi_volume_rate(last_bi, pre_bi)
    #     return {"lastBipre_volume_rate": lastBipre_volume_rate}

    @register_feature("last")
    def last_is_BiAndSeg_bsp_feature(self):
        """最后一个分段买卖点的类型特征"""
        if (not self.last_seg_bsp) or (not self.last_bsp):
            return None
        last_is_BiAndSeg_bsp = self.chan_is_BiAndSeg_bsp(self.last_bsp, self.last_seg_bsp)
        return {"last_is_BiAndSeg_bsp": last_is_BiAndSeg_bsp}
    @register_feature("pre")
    def pre_is_BiAndSeg_bsp_feature(self):
        """前一个分段买卖点的类型特征"""
        if (not self.pre_seg_bsp) or (not self.pre_bsp):
            return None
        pre_is_BiAndSeg_bsp = self.chan_is_BiAndSeg_bsp(self.pre_bsp, self.pre_seg_bsp)
        return {"pre_is_BiAndSeg_bsp": pre_is_BiAndSeg_bsp}

    @register_feature("last")
    def last_bsp_type_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        last_bsp_type = self.chan_bsp_type(self.last_bsp)
        return {"last_bsp_type": last_bsp_type}
    @register_feature("pre")
    def pre_bsp_type_feature(self):
        """当前买卖点的类型特征"""
        if not self.pre_bsp:
            return None
        pre_bsp_type = self.chan_bsp_type(self.pre_bsp)
        return {"pre_bsp_type": pre_bsp_type}
    @register_feature("last")
    def last_seg_bsp_type_feature(self):
        """最后一个分段买卖点的类型特征"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_type = self.chan_bsp_type(self.last_seg_bsp)
        return {"last_seg_bsp_type": last_seg_bsp_type}
    @register_feature("pre")
    def pre_seg_bsp_type_feature(self):
        """前一个分段买卖点的类型特征"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_type = self.chan_bsp_type(self.pre_seg_bsp)
        return {"pre_seg_bsp_type": pre_seg_bsp_type}

    @register_feature("last")
    def last_bsp_isB1_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_isB1 = self.chan_bsp_isB1(self.last_bsp)
        return {"bsp_isB1": bsp_isB1}
    @register_feature("last")
    def last_bsp_isB1p_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_isB1p = self.chan_bsp_isB1p(self.last_bsp)
        return {"bsp_isB1p": bsp_isB1p}
    @register_feature("last")
    def last_bsp_isS1_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_isS1 = self.chan_bsp_isS1(self.last_bsp)
        return {"bsp_isS1": bsp_isS1}
    @register_feature("last")
    def last_bsp_isS1p_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_isS1p = self.chan_bsp_isS1p(self.last_bsp)
        return {"bsp_isS1p": bsp_isS1p}
    @register_feature("pre")
    def pre_bsp_isB1_feature(self):
        """当前买卖点的类型特征"""
        if not self.pre_bsp:
            return None
        bsp_isB1 = self.chan_bsp_isB1(self.pre_bsp)
        return {"bsp_isB1": bsp_isB1}
    @register_feature("pre")
    def pre_bsp_isB1p_feature(self):
        """当前买卖点的类型特征"""
        if not self.pre_bsp:
            return None
        bsp_isB1p = self.chan_bsp_isB1p(self.pre_bsp)
        return {"bsp_isB1p": bsp_isB1p}
    @register_feature("pre")
    def pre_bsp_isS1_feature(self):
        """当前买卖点的类型特征"""
        if not self.pre_bsp:
            return None
        bsp_isS1 = self.chan_bsp_isS1(self.pre_bsp)
        return {"bsp_isS1": bsp_isS1}
    @register_feature("pre")
    def pre_bsp_isS1p_feature(self):
        """当前买卖点的类型特征"""
        if not self.pre_bsp:
            return None
        bsp_isS1p = self.chan_bsp_isS1p(self.pre_bsp)
        return {"bsp_isS1p": bsp_isS1p}
    @register_feature("last")
    def last_seg_bsp_isB1_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_isB1 = self.chan_bsp_isB1(self.last_seg_bsp)
        return {"last_seg_bsp_isB1": last_seg_bsp_isB1}
    @register_feature("last")
    def last_seg_bsp_isB1p_feature(self):
        """当前分段买卖点的类型特征"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_isB1p = self.chan_bsp_isB1p(self.last_seg_bsp)
        return {"last_seg_bsp_isB1p": last_seg_bsp_isB1p}
    @register_feature("last")
    def last_seg_bsp_isS1_feature(self):
        """当前分段买卖点的类型特征"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_isS1 = self.chan_bsp_isS1(self.last_seg_bsp)
        return {"last_seg_bsp_isS1": last_seg_bsp_isS1}
    @register_feature("last")
    def last_seg_bsp_isS1p_feature(self):
        """当前分段买卖点的类型特征"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_isS1p = self.chan_bsp_isS1p(self.last_seg_bsp)
        return {"last_seg_bsp_isS1p": last_seg_bsp_isS1p}
    @register_feature("pre")
    def pre_seg_bsp_isB1_feature(self):
        """前一个分段买卖点的类型特征"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_isB1 = self.chan_bsp_isB1(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB1": pre_seg_bsp_isB1}
    @register_feature("pre")
    def pre_seg_bsp_isB1p_feature(self):
        """前一个分段买卖点的类型特征"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_isB1p = self.chan_bsp_isB1p(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB1p": pre_seg_bsp_isB1p}
    @register_feature("pre")
    def pre_seg_bsp_isS1_feature(self):
        """前一个分段买卖点的类型特征"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_isS1 = self.chan_bsp_isS1(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS1": pre_seg_bsp_isS1}
    @register_feature("pre")
    def pre_seg_bsp_isS1p_feature(self):
        """前一个分段买卖点的类型特征"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_isS1p = self.chan_bsp_isS1p(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS1p": pre_seg_bsp_isS1p}
    
    # 2类特征
    # last_bsp 特征
    @register_feature("last")
    def last_bsp_isB2_feature(self):
        """当前买卖点的B2特征"""
        if not self.last_bsp:
            return None
        bsp_isB2 = self.chan_bsp_isB2(self.last_bsp)
        return {"bsp_isB2": bsp_isB2}
    @register_feature("last")
    def last_bsp_isS2_feature(self):
        """当前买卖点的S2特征"""
        if not self.last_bsp:
            return None
        bsp_isS2 = self.chan_bsp_isS2(self.last_bsp)
        return {"bsp_isS2": bsp_isS2}
    @register_feature("last")
    def last_bsp_isB2s_feature(self):
        """当前买卖点的B2s特征"""
        if not self.last_bsp:
            return None
        bsp_isB2s = self.chan_bsp_isB2s(self.last_bsp)
        return {"bsp_isB2s": bsp_isB2s}
    @register_feature("last")
    def last_bsp_isS2s_feature(self):
        """当前买卖点的S2s特征"""
        if not self.last_bsp:
            return None
        bsp_isS2s = self.chan_bsp_isS2s(self.last_bsp)
        return {"bsp_isS2s": bsp_isS2s}
    # pre_bsp 特征
    @register_feature("pre")
    def pre_bsp_isB2_feature(self):
        """前一个买卖点的B2特征"""
        if not self.pre_bsp:
            return None
        bsp_isB2 = self.chan_bsp_isB2(self.pre_bsp)
        return {"bsp_isB2": bsp_isB2}
    @register_feature("pre")
    def pre_bsp_isS2_feature(self):
        """前一个买卖点的S2特征"""
        if not self.pre_bsp:
            return None
        bsp_isS2 = self.chan_bsp_isS2(self.pre_bsp)
        return {"bsp_isS2": bsp_isS2}
    @register_feature("pre")
    def pre_bsp_isB2s_feature(self):
        """前一个买卖点的B2s特征"""
        if not self.pre_bsp:
            return None
        bsp_isB2s = self.chan_bsp_isB2s(self.pre_bsp)
        return {"bsp_isB2s": bsp_isB2s}
    @register_feature("pre")
    def pre_bsp_isS2s_feature(self):
        """前一个买卖点的S2s特征"""
        if not self.pre_bsp:
            return None
        bsp_isS2s = self.chan_bsp_isS2s(self.pre_bsp)
        return {"bsp_isS2s": bsp_isS2s}
    # last_seg_bsp 特征
    @register_feature("last")
    def last_seg_bsp_isB2_feature(self):
        """当前分段买卖点的B2特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isB2 = self.chan_bsp_isB2(self.last_seg_bsp)
        return {"last_seg_bsp_isB2": bsp_isB2}
    @register_feature("last")
    def last_seg_bsp_isS2_feature(self):
        """当前分段买卖点的S2特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isS2 = self.chan_bsp_isS2(self.last_seg_bsp)
        return {"last_seg_bsp_isS2": bsp_isS2}
    @register_feature("last")
    def last_seg_bsp_isB2s_feature(self):
        """当前分段买卖点的B2s特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isB2s = self.chan_bsp_isB2s(self.last_seg_bsp)
        return {"last_seg_bsp_isB2s": bsp_isB2s}
    @register_feature("last")
    def last_seg_bsp_isS2s_feature(self):
        """当前分段买卖点的S2s特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isS2s = self.chan_bsp_isS2s(self.last_seg_bsp)
        return {"last_seg_bsp_isS2s": bsp_isS2s}
    # pre_seg_bsp 特征
    @register_feature("pre")
    def pre_seg_bsp_isB2_feature(self):
        """前一个分段买卖点的B2特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isB2 = self.chan_bsp_isB2(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB2": bsp_isB2}
    @register_feature("pre")
    def pre_seg_bsp_isS2_feature(self):
        """前一个分段买卖点的S2特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isS2 = self.chan_bsp_isS2(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS2": bsp_isS2}
    @register_feature("pre")
    def pre_seg_bsp_isB2s_feature(self):
        """前一个分段买卖点的B2s特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isB2s = self.chan_bsp_isB2s(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB2s": bsp_isB2s}
    @register_feature("pre")
    def pre_seg_bsp_isS2s_feature(self):
        """前一个分段买卖点的S2s特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isS2s = self.chan_bsp_isS2s(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS2s": bsp_isS2s}

    # 3类特征
    # last_bsp 特征
    @register_feature("last")
    def last_bsp_isB3a_feature(self):
        """当前买卖点的B3a特征"""
        if not self.last_bsp:
            return None
        bsp_isB3a = self.chan_bsp_isB3a(self.last_bsp)
        return {"bsp_isB3a": bsp_isB3a}
    @register_feature("last")
    def last_bsp_isS3a_feature(self):
        """当前买卖点的S3a特征"""
        if not self.last_bsp:
            return None
        bsp_isS3a = self.chan_bsp_isS3a(self.last_bsp)
        return {"bsp_isS3a": bsp_isS3a}
    @register_feature("last")
    def last_bsp_isB3b_feature(self):
        """当前买卖点的B3b特征"""
        if not self.last_bsp:
            return None
        bsp_isB3b = self.chan_bsp_isB3b(self.last_bsp)
        return {"bsp_isB3b": bsp_isB3b}
    @register_feature("last")
    def last_bsp_isS3b_feature(self):
        """当前买卖点的S3b特征"""
        if not self.last_bsp:
            return None
        bsp_isS3b = self.chan_bsp_isS3b(self.last_bsp)
        return {"bsp_isS3b": bsp_isS3b}
    # pre_bsp 特征
    @register_feature("pre")
    def pre_bsp_isB3a_feature(self):
        """前一个买卖点的B3a特征"""
        if not self.pre_bsp:
            return None
        bsp_isB3a = self.chan_bsp_isB3a(self.pre_bsp)
        return {"pre_bsp_isB3a": bsp_isB3a}
    @register_feature("pre")
    def pre_bsp_isS3a_feature(self):
        """前一个买卖点的S3a特征"""
        if not self.pre_bsp:
            return None
        bsp_isS3a = self.chan_bsp_isS3a(self.pre_bsp)
        return {"pre_bsp_isS3a": bsp_isS3a}
    @register_feature("pre")
    def pre_bsp_isB3b_feature(self):
        """前一个买卖点的B3b特征"""
        if not self.pre_bsp:
            return None
        bsp_isB3b = self.chan_bsp_isB3b(self.pre_bsp)
        return {"pre_bsp_isB3b": bsp_isB3b}
    @register_feature("pre")
    def pre_bsp_isS3b_feature(self):
        """前一个买卖点的S3b特征"""
        if not self.pre_bsp:
            return None
        bsp_isS3b = self.chan_bsp_isS3b(self.pre_bsp)
        return {"pre_bsp_isS3b": bsp_isS3b}
    # last_seg_bsp 特征
    @register_feature("last")
    def last_seg_bsp_isB3a_feature(self):
        """当前分段买卖点的B3a特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isB3a = self.chan_bsp_isB3a(self.last_seg_bsp)
        return {"last_seg_bsp_isB3a": bsp_isB3a}
    @register_feature("last")
    def last_seg_bsp_isS3a_feature(self):
        """当前分段买卖点的S3a特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isS3a = self.chan_bsp_isS3a(self.last_seg_bsp)
        return {"last_seg_bsp_isS3a": bsp_isS3a}
    @register_feature("last")
    def last_seg_bsp_isB3b_feature(self):
        """当前分段买卖点的B3b特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isB3b = self.chan_bsp_isB3b(self.last_seg_bsp)
        return {"last_seg_bsp_isB3b": bsp_isB3b}
    @register_feature("last")
    def last_seg_bsp_isS3b_feature(self):
        """当前分段买卖点的S3b特征"""
        if not self.last_seg_bsp:
            return None
        bsp_isS3b = self.chan_bsp_isS3b(self.last_seg_bsp)
        return {"last_seg_bsp_isS3b": bsp_isS3b}
    # pre_seg_bsp 特征
    @register_feature("pre")
    def pre_seg_bsp_isB3a_feature(self):
        """前一个分段买卖点的B3a特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isB3a = self.chan_bsp_isB3a(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB3a": bsp_isB3a}
    @register_feature("pre")
    def pre_seg_bsp_isS3a_feature(self):
        """前一个分段买卖点的S3a特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isS3a = self.chan_bsp_isS3a(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS3a": bsp_isS3a}
    @register_feature("pre")
    def pre_seg_bsp_isB3b_feature(self):
        """前一个分段买卖点的B3b特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isB3b = self.chan_bsp_isB3b(self.pre_seg_bsp)
        return {"pre_seg_bsp_isB3b": bsp_isB3b}
    @register_feature("pre")
    def pre_seg_bsp_isS3b_feature(self):
        """前一个分段买卖点的S3b特征"""
        if not self.pre_seg_bsp:
            return None
        bsp_isS3b = self.chan_bsp_isS3b(self.pre_seg_bsp)
        return {"pre_seg_bsp_isS3b": bsp_isS3b}

    # last_bsp 特征
    @register_feature("last")
    def last_bsp_is_buy_feature(self):
        """当前买卖点是否为买入"""
        if not self.last_bsp:
            return None
        last_bsp_is_buy = self.last_bsp.is_buy
        return {"last_bsp_is_buy": last_bsp_is_buy}
    @register_feature("last")
    def last_bsp_is_sell_feature(self):
        """当前买卖点是否为卖出"""
        if not self.last_bsp:
            return None
        last_bsp_is_sell = not self.last_bsp.is_buy
        return {"last_bsp_is_sell": last_bsp_is_sell}
    @register_feature("last")
    def last_bsp_BuyOrSell_feature(self):
        """当前买卖点的买卖方向"""
        if not self.last_bsp:
            return None
        is_buy = self.last_bsp.is_buy
        bsp_BuyOrSell = 1 if is_buy else -1
        return {"bsp_BuyOrSell": bsp_BuyOrSell}
    # pre_bsp 特征
    @register_feature("pre")
    def pre_bsp_is_buy_feature(self):
        """前一个买卖点是否为买入"""
        if not self.pre_bsp:
            return None
        pre_bsp_is_buy = self.pre_bsp.is_buy
        return {"pre_bsp_is_buy": pre_bsp_is_buy}
    @register_feature("pre")
    def pre_bsp_is_sell_feature(self):
        """前一个买卖点是否为卖出"""
        if not self.pre_bsp:
            return None
        pre_bsp_is_sell = not self.pre_bsp.is_buy
        return {"pre_bsp_is_sell": pre_bsp_is_sell}
    @register_feature("pre")
    def pre_bsp_BuyOrSell_feature(self):
        """前一个买卖点的买卖方向"""
        if not self.pre_bsp:
            return None
        is_buy = self.pre_bsp.is_buy
        pre_bsp_BuyOrSell = 1 if is_buy else -1
        return {"pre_bsp_BuyOrSell": pre_bsp_BuyOrSell}
    # last_seg_bsp 特征
    @register_feature("last")
    def last_seg_bsp_is_buy_feature(self):
        """当前分段买卖点是否为买入"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_is_buy = self.last_seg_bsp.is_buy
        return {"last_seg_bsp_is_buy": last_seg_bsp_is_buy}
    @register_feature("last")
    def last_seg_bsp_is_sell_feature(self):
        """当前分段买卖点是否为卖出"""
        if not self.last_seg_bsp:
            return None
        last_seg_bsp_is_sell = not self.last_seg_bsp.is_buy
        return {"last_seg_bsp_is_sell": last_seg_bsp_is_sell}
    @register_feature("last")
    def last_seg_bsp_BuyOrSell_feature(self):
        """当前分段买卖点的买卖方向"""
        if not self.last_seg_bsp:
            return None
        is_buy = self.last_seg_bsp.is_buy
        last_seg_bsp_BuyOrSell = 1 if is_buy else -1
        return {"last_seg_bsp_BuyOrSell": last_seg_bsp_BuyOrSell}
    # pre_seg_bsp 特征
    @register_feature("pre")
    def pre_seg_bsp_is_buy_feature(self):
        """前一个分段买卖点是否为买入"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_is_buy = self.pre_seg_bsp.is_buy
        return {"pre_seg_bsp_is_buy": pre_seg_bsp_is_buy}
    @register_feature("pre")
    def pre_seg_bsp_is_sell_feature(self):
        """前一个分段买卖点是否为卖出"""
        if not self.pre_seg_bsp:
            return None
        pre_seg_bsp_is_sell = not self.pre_seg_bsp.is_buy
        return {"pre_seg_bsp_is_sell": pre_seg_bsp_is_sell}
    @register_feature("pre")
    def pre_seg_bsp_BuyOrSell_feature(self):
        """前一个分段买卖点的买卖方向"""
        if not self.pre_seg_bsp:
            return None
        is_buy = self.pre_seg_bsp.is_buy
        pre_seg_bsp_BuyOrSell = 1 if is_buy else -1
        return {"pre_seg_bsp_BuyOrSell": pre_seg_bsp_BuyOrSell}

    # 待删除特征
    @register_feature("chan")
    def last_bsp_is1t_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1t = bsp_type == "1" or bsp_type == "1p"
        return {"bsp_is1t": bsp_is1t}
    @register_feature("chan")
    def last_bsp_is2t_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2t = bsp_type == "2" or bsp_type == "2s"
        return {"bsp_is2t": bsp_is2t}
    @register_feature("chan")
    def last_bsp_is3t_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3t = bsp_type == "3a" or bsp_type == "3b"
        return {"bsp_is3t": bsp_is3t}
    @register_feature("chan")
    def last_bsp_is1_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1 = bsp_type == "1"
        return {"bsp_is1": bsp_is1}
    @register_feature("chan")
    def last_bsp_is2_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2 = bsp_type == "2"
        return {"bsp_is2": bsp_is2}
    @register_feature("chan")
    def last_bsp_is1p_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is1p = bsp_type == "1p"
        return {"bsp_is1p": bsp_is1p}
    @register_feature("chan")
    def last_bsp_is2s_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is2s = bsp_type == "2s"
        return {"bsp_is2s": bsp_is2s}
    @register_feature("chan")
    def last_bsp_is3a_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3a = bsp_type == "3a"
        return {"bsp_is3a": bsp_is3a}
    @register_feature("chan")
    def last_bsp_is3b_feature(self):
        """当前买卖点的类型特征"""
        if not self.last_bsp:
            return None
        bsp_type = self.last_bsp.type[0].value  # 获取当前买卖点的类型
        bsp_is3b = bsp_type == "3b"
        return {"bsp_is3b": bsp_is3b}
    
